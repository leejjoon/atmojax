# The atmojax weights file (`.npz`)

`atmojax.load(path)` / `Weights.load(path)` read one plain NumPy `.npz`. `atmojax-export` writes it
from a Payne Zero checkpoint, but any code that trains a network of the same shape can write it
directly (for example an emulator trained on ATLAS12 models) and atmojax will evaluate it, with
`grad`/`jacfwd`/`jit`/`vmap`, without changes. This page is the contract such a writer has to meet.
Constructing a `Weights` (which `load` does) checks the feature names, every shape, the layer
chaining, `tau`, and the signs of the scales, and raises `ValueError` listing every violation;
`Weights.problems()` returns the same list without raising. It cannot check the activation, so a
network that wasn't SiLU-between-Linear-layers still loads and gives wrong output.

## The forward pass the file parameterizes

```
features  f = (5040/Teff, label_1, ..., label_{n-1})                     # (n,)  feature units
x         = (f - label_mean) / label_std
x         = silu(x @ W0.T + b0) ... silu(x @ W{L-2}.T + b{L-2})
y         = x @ W{L-1}.T + b{L-1}                                          # (k,)  no activation
coeffs    = y * pca_coefficient_std + pca_coefficient_mean                 # (k,)
flat      = (coeffs @ pca_basis) * pca_coordinate_std + pca_coordinate_mean   # (480,)
c         = flat.reshape(80, 6)                                            # row-major: flat[6*j + i] = c[j, i]
```

The MLP runs in `mlp_dtype` (float32 by default); the PCA decode after it runs in the dtype of the
stored arrays (float64 when `jax_enable_x64` is on).
`c` holds six coordinate fields per layer `j` (outermost first), on the fixed grid
`log10 tau_Ross[j] = -6.875 + 0.125 j`, `j = 0..79` (`atmojax.LOG_TAU_ROSS`). They decode to physical
CGS fields as follows (`atmojax.FIELDS` order), and a trainer computes its targets by the inverse:

| `i` | coordinate `c[:, i]` | decoded field | inverse (training target) |
|---|---|---|---|
| 0 | log10 column-mass increment | `m = cumsum(10**c0)` [g cm⁻²] | `c0[j] = log10(m[j] - m[j-1])`, with `m[-1] = 0` |
| 1 | log10 T relative to grey | `T = Teff (0.75 (tau + 2/3))**0.25 * 10**c1` [K] | `c1 = log10(T / T_grey)` |
| 2 | log10 gas pressure | `P_gas = 10**c2` [dyn cm⁻²] | `log10 P_gas` |
| 3 | log10 electron density | `n_e = 10**c3` [cm⁻³] | `log10 n_e` |
| 4 | log10 Rosseland opacity | `kappa_R = 10**c4` [cm² g⁻¹] | `log10 kappa_R` |
| 5 | asinh radiative acceleration | `g_rad = acceleration_scale * sinh(c5)` [cm s⁻²] | `asinh(g_rad / acceleration_scale)` |

`m` must increase strictly with depth, which the `cumsum` guarantees. `T_grey` uses the file's `tau`
and the Teff label in K. For ATLAS-style decks these are the `RHOX`, `T`, `P`, `XNE`, `ABROSS` and
`ACCRAD` columns, after interpolation onto the grid above.

## Keys

`n` = number of labels, `k` = number of PCA components, `L` = number of Linear layers.

| key | dtype, shape | meaning |
|---|---|---|
| `family` | str, `()` | free-form model name, shown in error messages (`"five_label"`, `"cno8"`, ...) |
| `source_format` | str, `()` | optional provenance tag; not read by `load`. Exported files carry the Payne Zero checkpoint format. Other writers should name their own source here (e.g. `"atlas12_a0v_v1"`) |
| `feature_fields` | str, `(n,)` | the network's input features, in order; each must be a key of `atmojax.initializer.LABEL_OF_FEATURE` |
| `label_mean`, `label_std` | float, `(n,)` | input standardization, in feature units |
| `label_bounds` | float, `(n, 2)` | training box `[lo, hi]` per feature, in feature units; used only by `in_support` |
| `tau` | float, `(80,)` | `10**LOG_TAU_ROSS`; used for `T_grey`, and the layers are assumed to sit on this grid |
| `acceleration_scale` | float, `()` | the `g_rad` scale above, > 0 |
| `pca_coordinate_mean`, `pca_coordinate_std` | float, `(480,)` | per-coordinate standardization of `flat` |
| `pca_basis` | float, `(k, 480)` | decoder rows; they need not be orthonormal (any linear decoder works) |
| `pca_coefficient_mean`, `pca_coefficient_std` | float, `(k,)` | de-standardization of the network output |
| `W0` ... `W{L-1}` | float, `(out, in)` | Linear weights in Torch layout; `W0` has `in = n`, `W{L-1}` has `out = k` |
| `b0` ... `b{L-1}` | float, `(out,)` | Linear biases |
| `n_layers` | int, `()` | `L`, the number of Linear layers (not atmosphere layers) |

## Rules a writer must follow

- **Teff comes first, as `5040/Teff`.** `feature_fields[0]` must be
  `temperature_ratio_5040_k_over_temperature`. atmojax converts label 0 from K to `5040/Teff` before
  the network and uses it in K for `T_grey`. The other features enter the network as given.
- **Only known feature names.** Labels are named via `LABEL_OF_FEATURE` (`teff`, `logg`, `m_h`,
  `alpha_m`, `vmic`, `c_m`, `n_m`, `o_m`). Any subset in any order after Teff is fine. A new label
  needs a new entry there first. The order of `feature_fields` sets the label order of `labels()`
  and the last axis of `log_state_jacobian`.
- **Strings as NumPy unicode arrays** (`np.array("five_label")`, `np.array([...])`), never object
  arrays: the file is read with `allow_pickle=False`.
- **float64 recommended** for every float array. The loader keeps dtypes as stored and casts the MLP
  weights to `mlp_dtype`.
- **SiLU between Linear layers, none after the last.** Any depth and widths are fine. Other
  activations, normalization layers or skip connections are not representable.

## Minimal writer

```python
import numpy as np
from atmojax import LOG_TAU_ROSS

arrays = {
    "family": np.array("a0v"), "source_format": np.array("atlas12_a0v_v1"),
    "feature_fields": np.array(["temperature_ratio_5040_k_over_temperature", "log10_surface_gravity_cgs",
                                "metallicity", "microturbulence_km_s"]),
    "label_mean": label_mean, "label_std": label_std, "label_bounds": label_bounds,   # feature units
    "tau": 10.0 ** LOG_TAU_ROSS, "acceleration_scale": np.float64(acceleration_scale),
    "pca_coordinate_mean": cmean, "pca_coordinate_std": cstd, "pca_basis": basis,
    "pca_coefficient_mean": kmean, "pca_coefficient_std": kstd,
    "n_layers": np.int64(len(layers)),
}
for i, (W, b) in enumerate(layers):           # W: (out, in), b: (out,)
    arrays[f"W{i}"], arrays[f"b{i}"] = W, b
np.savez("a0v.npz", **arrays)
```

`tests/test_initializer.py::test_weights_roundtrip` writes a file this way from synthetic weights and
checks that it reproduces the in-memory model exactly.
