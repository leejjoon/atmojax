"""Differentiable atmosphere initializer: stellar labels -> an 80-layer 1D LTE atmosphere, in JAX.

A JAX re-implementation of the forward pass of Payne Zero's released atmosphere initializers
(``payne_zero_atmosphere.warm_start.AtmosphereInitializer.predict``): a SiLU MLP predicts PCA
coefficients that decode to six fields on the fixed Rosseland grid log10 tau = -6.875 + 0.125 j.

Two things of the original are deliberately absent, so ``jax.grad`` / ``jax.jacfwd`` work:
the ``np.clip`` guards (inactive inside the training support) and Payne Zero's fixed-digit text-deck
round trip (a staircase with zero derivative almost everywhere).  Label support is checked with
:meth:`AtmosphereInitializer.in_support`, outside the traced computation.

Weights are not shipped; export them once from a Payne Zero install with ``atmojax-export``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

LAYERS = 80
LOG_TAU_ROSS = -6.875 + 0.125 * np.arange(LAYERS)

#: Decoded output fields, in order (CGS: g cm^-2, K, dyn cm^-2, cm^-3, cm^2 g^-1, cm s^-2).
FIELDS = ("column_mass", "temperature", "gas_pressure", "electron_density",
          "rosseland_opacity", "radiative_acceleration")

#: Public label names, mapped from the checkpoint's feature fields.
LABEL_OF_FEATURE = {
    "temperature_ratio_5040_k_over_temperature": "teff",
    "log10_surface_gravity_cgs": "logg",
    "metallicity": "m_h",
    "alpha_enhancement": "alpha_m",
    "microturbulence_km_s": "vmic",
    "carbon_enhancement": "c_m",
    "nitrogen_enhancement": "n_m",
    "oxygen_enhancement": "o_m",
}


@dataclass(frozen=True)
class Weights:
    """Everything the forward pass needs, as float64 NumPy arrays (see :mod:`atmojax.export`)."""

    family: str
    feature_fields: tuple[str, ...]
    label_mean: np.ndarray
    label_std: np.ndarray
    label_bounds: np.ndarray          # (n_labels, 2), in feature units (Teff as 5040/Teff)
    tau: np.ndarray                   # (80,) standard Rosseland optical depth
    acceleration_scale: float
    pca_coordinate_mean: np.ndarray
    pca_coordinate_std: np.ndarray
    pca_basis: np.ndarray             # (n_components, 480)
    pca_coefficient_mean: np.ndarray
    pca_coefficient_std: np.ndarray
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]   # (W, b) per Linear, torch layout (out, in)

    def __post_init__(self):
        problems = self.problems()
        if problems:
            raise ValueError("invalid atmojax weights (see docs/weights-format.md):\n  " + "\n  ".join(problems))

    def problems(self) -> list[str]:
        """Violations of the weights-file contract (docs/weights-format.md); empty when valid."""
        out = []
        f = self.feature_fields
        if not f or f[0] != "temperature_ratio_5040_k_over_temperature":
            out.append(f"feature_fields[0] must be 'temperature_ratio_5040_k_over_temperature', got {f[:1]}")
        if unknown := [x for x in f if x not in LABEL_OF_FEATURE]:
            out.append(f"unknown feature_fields {unknown}; add them to LABEL_OF_FEATURE first")
        if len(set(f)) != len(f):
            out.append(f"duplicate feature_fields {f}")
        n, size = len(f), LAYERS * len(FIELDS)
        k = np.shape(self.pca_basis)[0] if np.ndim(self.pca_basis) == 2 else None
        for name, shape in [("label_mean", (n,)), ("label_std", (n,)), ("label_bounds", (n, 2)),
                            ("tau", (LAYERS,)), ("pca_coordinate_mean", (size,)), ("pca_coordinate_std", (size,)),
                            ("pca_basis", (k, size)), ("pca_coefficient_mean", (k,)), ("pca_coefficient_std", (k,))]:
            if np.shape(getattr(self, name)) != shape:
                out.append(f"{name} has shape {np.shape(getattr(self, name))}, expected {shape}")
        if out:
            return out
        if np.any(self.label_std <= 0) or np.any(self.pca_coordinate_std <= 0) or np.any(self.pca_coefficient_std <= 0):
            out.append("label_std, pca_coordinate_std and pca_coefficient_std must be > 0")
        if np.any(self.label_bounds[:, 0] > self.label_bounds[:, 1]):
            out.append("label_bounds must be [lo, hi] with lo <= hi")
        if not np.allclose(self.tau, 10.0 ** LOG_TAU_ROSS, rtol=1e-6, atol=0.0):
            out.append("tau must be 10**LOG_TAU_ROSS (log10 tau = -6.875 + 0.125 j)")
        if not self.acceleration_scale > 0:
            out.append(f"acceleration_scale must be > 0, got {self.acceleration_scale}")
        if not self.layers:
            out.append("no Linear layers")
        width = n
        for i, (W, b) in enumerate(self.layers):
            if np.ndim(W) != 2 or np.shape(W)[1] != width or np.shape(b) != (np.shape(W)[0],):
                out.append(f"layer {i}: W {np.shape(W)} / b {np.shape(b)} do not chain from width {width} "
                           f"(expected W (out, {width}), b (out,))")
                return out
            width = np.shape(W)[0]
        if self.layers and width != k:
            out.append(f"last layer outputs {width}, but pca_basis has {k} components")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Weights":
        z = np.load(path, allow_pickle=False)
        n = int(z["n_layers"])
        return cls(
            family=str(z["family"]),
            feature_fields=tuple(str(s) for s in z["feature_fields"]),
            label_mean=z["label_mean"], label_std=z["label_std"], label_bounds=z["label_bounds"],
            tau=z["tau"], acceleration_scale=float(z["acceleration_scale"]),
            pca_coordinate_mean=z["pca_coordinate_mean"], pca_coordinate_std=z["pca_coordinate_std"],
            pca_basis=z["pca_basis"], pca_coefficient_mean=z["pca_coefficient_mean"],
            pca_coefficient_std=z["pca_coefficient_std"],
            layers=tuple((z[f"W{i}"], z[f"b{i}"]) for i in range(n)),
        )


class AtmosphereInitializer:
    """Labels -> atmosphere, differentiable.

    ``labels`` is a length-``n_labels`` array ordered as :attr:`label_names` -- for the five-label
    family ``(teff, logg, m_h, alpha_m, vmic)``; the CNO family appends ``(c_m, n_m, o_m)``.
    ``teff`` is in K and ``vmic`` in km/s; the rest are dex.

    ``mlp_dtype`` defaults to float32, matching the original Torch evaluation (parity ~1e-6);
    pass ``jnp.float64`` for smoother high-order derivatives (requires ``jax_enable_x64``).
    """

    def __init__(self, weights: Weights | str | Path, mlp_dtype=jnp.float32):
        w = weights if isinstance(weights, Weights) else Weights.load(weights)
        self.weights = w
        self.family = w.family
        self.label_names = tuple(LABEL_OF_FEATURE[f] for f in w.feature_fields)
        self.mlp_dtype = mlp_dtype
        self._p = {
            "mean": jnp.asarray(w.label_mean), "std": jnp.asarray(w.label_std), "tau": jnp.asarray(w.tau),
            "cmean": jnp.asarray(w.pca_coordinate_mean), "cstd": jnp.asarray(w.pca_coordinate_std),
            "basis": jnp.asarray(w.pca_basis),
            "kmean": jnp.asarray(w.pca_coefficient_mean), "kstd": jnp.asarray(w.pca_coefficient_std),
            "layers": [(jnp.asarray(W.T, mlp_dtype), jnp.asarray(b, mlp_dtype)) for W, b in w.layers],
        }

    @property
    def n_labels(self) -> int:
        return len(self.label_names)

    def labels(self, **values: float) -> jnp.ndarray:
        """Build the label vector from keywords, e.g. ``labels(teff=5777, logg=4.44, m_h=0, alpha_m=0, vmic=1)``."""
        missing = [n for n in self.label_names if n not in values]
        extra = [n for n in values if n not in self.label_names]
        if missing or extra:
            raise ValueError(f"{self.family} needs {self.label_names}; missing {missing}, unexpected {extra}")
        return jnp.asarray([values[n] for n in self.label_names], dtype=jnp.result_type(float))

    def _features(self, labels):
        return jnp.concatenate([jnp.atleast_1d(5040.0 / labels[0]), labels[1:]])

    def in_support(self, labels, tolerance: float = 0.0) -> bool:
        """Whether the labels lie inside the training box (not traceable; call outside jit/grad)."""
        f = np.asarray(self._features(jnp.asarray(labels, dtype=float)))
        lo, hi = self.weights.label_bounds[:, 0], self.weights.label_bounds[:, 1]
        pad = tolerance * np.maximum(1.0, hi - lo)
        return bool(np.all((f >= lo - pad) & (f <= hi + pad)))

    def coordinates(self, labels) -> jnp.ndarray:
        """The network's raw (80, 6) coordinate fields (log-scale), before the physical decode."""
        p = self._p
        x = ((self._features(labels) - p["mean"]) / p["std"]).astype(self.mlp_dtype)
        for i, (W, b) in enumerate(p["layers"]):
            x = x @ W + b
            if i < len(p["layers"]) - 1:
                x = jax.nn.silu(x)
        coefficients = x.astype(p["kmean"].dtype) * p["kstd"] + p["kmean"]
        flat = (coefficients @ p["basis"]) * p["cstd"] + p["cmean"]
        return flat.reshape(LAYERS, len(FIELDS))

    def predict(self, labels) -> jnp.ndarray:
        """(80, 6) physical fields, ordered as :data:`FIELDS`, outermost layer first."""
        c = self.coordinates(labels)
        grey = labels[0] * (0.75 * (self._p["tau"] + 2.0 / 3.0)) ** 0.25
        return jnp.stack([
            jnp.cumsum(10.0 ** c[:, 0]),
            grey * 10.0 ** c[:, 1],
            10.0 ** c[:, 2],
            10.0 ** c[:, 3],
            10.0 ** c[:, 4],
            self.weights.acceleration_scale * jnp.sinh(c[:, 5]),
        ], axis=1)

    def predict_dict(self, labels) -> dict[str, jnp.ndarray]:
        out = self.predict(labels)
        return {name: out[:, i] for i, name in enumerate(FIELDS)}

    def state(self, labels, fields: Sequence[str] = ("temperature", "gas_pressure", "column_mass",
                                                     "electron_density")) -> jnp.ndarray:
        """Selected fields stacked as (len(fields), 80).  The default order (T, P_gas, m, n_e) is the
        atmosphere state used by differentiable_stellar_spectroscopy's coupling seam."""
        out = self.predict(labels)
        return jnp.stack([out[:, FIELDS.index(f)] for f in fields])

    def log_state_jacobian(self, labels, fields: Sequence[str] = ("temperature", "gas_pressure",
                                                                  "column_mass", "electron_density")):
        """d ln(field) / d label at fixed optical depth: shape (len(fields), 80, n_labels)."""
        return jax.jacfwd(lambda l: jnp.log(self.state(l, fields)))(jnp.asarray(labels, dtype=float))


def load(path: str | Path, **kwargs) -> AtmosphereInitializer:
    """Load an initializer from an exported ``.npz`` (see ``atmojax-export``)."""
    return AtmosphereInitializer(path, **kwargs)
