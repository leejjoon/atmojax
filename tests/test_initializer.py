"""Tests.  The decode/derivative tests use small synthetic weights and always run; the parity tests
compare against Payne Zero's own implementation and run only when payne-zero, torch and its
checkpoints are available (set ATMOJAX_SKIP_PARITY=1 to skip them explicitly)."""
from __future__ import annotations

import dataclasses
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from atmojax import FIELDS, LOG_TAU_ROSS, AtmosphereInitializer, Weights  # noqa: E402

FIVE = ("temperature_ratio_5040_k_over_temperature", "log10_surface_gravity_cgs", "metallicity",
        "alpha_enhancement", "microturbulence_km_s")
SUN = np.array([5777.0, 4.44, 0.0, 0.0, 1.0])


def synthetic_weights(seed=0, width=16, n_comp=8) -> Weights:
    rng = np.random.default_rng(seed)
    sizes = [5, width, width, n_comp]
    layers = tuple((rng.normal(0, 0.3, (o, i)), rng.normal(0, 0.1, o)) for i, o in zip(sizes[:-1], sizes[1:]))
    cmean = np.tile([-3.0, 0.0, 4.0, 13.0, -1.0, 0.0], 80)    # log10 dm, log10 T/T_grey, log P, log n_e, ...
    return Weights(
        family="five_label", feature_fields=FIVE,
        label_mean=np.array([0.9, 3.0, -1.0, 0.2, 2.0]), label_std=np.array([0.2, 1.3, 0.8, 0.2, 1.0]),
        label_bounds=np.array([[0.48, 1.26], [0.7, 5.3], [-2.5, 0.5], [-0.1, 0.5], [0.5, 4.0]]),
        tau=10.0 ** LOG_TAU_ROSS, acceleration_scale=2.0,
        pca_coordinate_mean=cmean, pca_coordinate_std=np.full(480, 0.05),
        pca_basis=rng.normal(0, 0.2, (n_comp, 480)),
        pca_coefficient_mean=np.zeros(n_comp), pca_coefficient_std=np.ones(n_comp), layers=layers)


@pytest.fixture(scope="module")
def init():
    return AtmosphereInitializer(synthetic_weights(), mlp_dtype=jnp.float64)


def test_shapes_and_physical_decode(init):
    out = np.asarray(init.predict(jnp.asarray(SUN)))
    assert out.shape == (80, len(FIELDS))
    assert np.all(np.diff(out[:, 0]) > 0), "column mass must increase with depth"
    assert np.all(out[:, :5] > 0)
    assert init.state(jnp.asarray(SUN)).shape == (4, 80)


def test_labels_helper_and_support(init):
    lab = init.labels(teff=5777, logg=4.44, m_h=0.0, alpha_m=0.0, vmic=1.0)
    np.testing.assert_allclose(lab, SUN)
    with pytest.raises(ValueError):
        init.labels(teff=5777)
    assert init.in_support(SUN)
    assert not init.in_support(np.array([3000.0, 4.44, 0.0, 0.0, 1.0]))


def test_jacobian_matches_finite_differences(init):
    J = np.asarray(init.log_state_jacobian(SUN))          # (4, 80, 5)
    assert J.shape == (4, 80, 5)
    f = lambda l: np.log(np.asarray(init.state(jnp.asarray(l))))
    steps = np.array([1.0, 1e-4, 1e-4, 1e-4, 1e-4])
    for i, h in enumerate(steps):
        e = np.zeros(5); e[i] = h
        fd = (f(SUN + e) - f(SUN - e)) / (2 * h)
        np.testing.assert_allclose(J[:, :, i], fd, rtol=1e-5, atol=1e-9)


def test_jit_and_grad_compose(init):
    loss = jax.jit(lambda l: jnp.sum(init.state(l)[0] ** 2))
    g = jax.grad(loss)(jnp.asarray(SUN))
    assert np.all(np.isfinite(np.asarray(g)))


def test_weights_roundtrip(tmp_path, init):
    w = init.weights
    arrays = {"family": np.array(w.family), "feature_fields": np.array(w.feature_fields),
              "label_mean": w.label_mean, "label_std": w.label_std, "label_bounds": w.label_bounds,
              "tau": w.tau, "acceleration_scale": np.float64(w.acceleration_scale),
              "pca_coordinate_mean": w.pca_coordinate_mean, "pca_coordinate_std": w.pca_coordinate_std,
              "pca_basis": w.pca_basis, "pca_coefficient_mean": w.pca_coefficient_mean,
              "pca_coefficient_std": w.pca_coefficient_std, "n_layers": np.int64(len(w.layers))}
    for i, (W, b) in enumerate(w.layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    np.savez(tmp_path / "w.npz", **arrays)
    again = AtmosphereInitializer(tmp_path / "w.npz", mlp_dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(again.predict(SUN)), np.asarray(init.predict(SUN)))


@pytest.mark.parametrize("change, message", [
    (lambda w: {"feature_fields": w.feature_fields[1:] + w.feature_fields[:1]}, "feature_fields[0]"),
    (lambda w: {"feature_fields": w.feature_fields[:4] + ("helium",)}, "unknown feature_fields"),
    (lambda w: {"label_std": w.label_std[:4]}, "label_std has shape"),
    (lambda w: {"tau": 10.0 ** (LOG_TAU_ROSS + 0.1)}, "tau must be"),
    (lambda w: {"pca_basis": w.pca_basis[:5]}, "pca_coefficient_mean has shape"),
    (lambda w: {"layers": ((w.layers[0][0].T, w.layers[0][1]),) + w.layers[1:]}, "do not chain"),
    (lambda w: {"layers": w.layers[:2]}, "last layer outputs"),
    (lambda w: {"acceleration_scale": 0.0}, "acceleration_scale"),
])
def test_invalid_weights_are_rejected(change, message):
    w = synthetic_weights()
    with pytest.raises(ValueError, match=re.escape(message)):
        dataclasses.replace(w, **change(w))


# --- parity with Payne Zero's own implementation -------------------------------------------------

PARITY = {
    "five_label": [(5777, 4.44, 0.0, 0.0, 1.0), (4286, 1.66, -0.52, 0.30, 1.7), (5000, 2.5, -1.5, 0.4, 1.5),
                   (9000, 4.2, 0.3, 0.0, 2.0), (4100, 1.0, -2.0, 0.4, 2.5)],
    "cno8": [(4600, 2.2, -0.4, 0.2, 1.5, -0.25, 0.35, 0.15), (5777, 4.44, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
             (4300, 1.5, -1.0, 0.3, 1.8, 0.4, -0.4, 0.3)],
}


def _parity_or_skip(family, tmp_path):
    if os.environ.get("ATMOJAX_SKIP_PARITY"):
        pytest.skip("ATMOJAX_SKIP_PARITY set")
    pytest.importorskip("torch")
    pz = pytest.importorskip("payne_zero_atmosphere.warm_start")
    from atmojax.export import default_checkpoint, export
    ckpt = default_checkpoint(family)
    if not ckpt.exists() or ckpt.stat().st_size < 10_000:       # missing, or a Git LFS pointer
        pytest.skip(f"{ckpt} not available")
    ref = pz.load_atmosphere_initializer(checkpoint_path=ckpt, device="cpu")
    return ref, AtmosphereInitializer(export(ckpt, tmp_path / f"{family}.npz"))


@pytest.mark.parametrize("family", ["five_label", "cno8"])
def test_parity_with_payne_zero(family, tmp_path):
    ref, init = _parity_or_skip(family, tmp_path)
    names = ("effective_temperature", "log_surface_gravity", "metallicity", "alpha_enhancement",
             "microturbulence_km_s", "carbon_enhancement", "nitrogen_enhancement", "oxygen_enhancement")
    predict = jax.jit(init.predict)
    for lab in PARITY[family]:
        r = ref.predict(**dict(zip(names, lab)))
        y = np.asarray(predict(jnp.asarray(lab, float)))
        for i, k in enumerate(FIELDS):
            if k == "radiative_acceleration":
                err = np.max(np.abs(y[:, i] - r[k])) / np.max(np.abs(r[k]))
                assert err < 1e-4, (family, lab, k, err)
            else:
                err = np.max(np.abs(y[:, i] / r[k] - 1))
                assert err < 2e-5, (family, lab, k, err)
