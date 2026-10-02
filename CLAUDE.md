# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`atmojax` is an independent JAX re-implementation of the forward pass of Payne Zero's released
atmosphere initializers: a SiLU MLP maps stellar labels (Teff, logg, [M/H], [alpha/M], vmic, and
optionally C/N/O) to PCA coefficients that decode to a complete 80-layer 1D LTE atmosphere (column
mass, temperature, gas pressure, electron density, Rosseland opacity, radiative acceleration) on the
fixed grid `log10 tau_Ross = -6.875 + 0.125*j`. It's a pure function, so `jax.grad`/`jax.jacfwd`/
`jax.jit`/`jax.vmap` all work through it — that differentiability is the entire point of the project
(see "What is deliberately left out" below).

Network weights are Payne Zero's and are never redistributed in this repo — they're exported once
from a local Payne Zero install via `atmojax-export`, then only JAX/NumPy are needed at runtime.

## Commands

```bash
pip install -e .                 # runtime: jax + numpy only
pip install -e ".[export]"       # + torch, needed once to export weights from Payne Zero
pip install -e ".[test]"         # + pytest
pip install -e ".[validation]"   # + torch, scipy, for validation/

pytest                           # run all tests
pytest tests/test_initializer.py::test_jacobian_matches_finite_differences   # single test
ATMOJAX_SKIP_PARITY=1 pytest     # skip the Payne Zero parity tests explicitly

atmojax-export five_label five_label.npz   # export weights (needs payne-zero + its Git-LFS checkpoints)
atmojax-export cno8 cno8.npz
atmojax-export --checkpoint path/to/checkpoint.pt out.npz
```

There is no linter/formatter config in this repo; match the existing style (dense, no docstring
boilerplate, type hints on public signatures).

## Architecture

### `src/atmojax/` — the package (jax + numpy only)

- `initializer.py` is the whole implementation:
  - `Weights` (frozen dataclass): everything the forward pass needs, loaded from an exported `.npz`.
  - `AtmosphereInitializer`: holds `Weights` as JAX arrays (`self._p` dict) plus `mlp_dtype`
    (float32 by default to match Payne Zero's own Torch evaluation; pass `jnp.float64` for smoother
    high-order derivatives, which requires `jax.config.update("jax_enable_x64", True)`).
    - `.labels(**kwargs)` builds the ordered label vector from keyword stellar parameters.
    - `.coordinates(labels)`: normalize -> MLP (SiLU between layers) -> de-normalize -> PCA basis
      expansion -> reshape to `(80, 6)` raw per-field coordinates.
    - `.predict(labels)`: physically decodes those coordinates per field — column mass via
      `cumsum(10**c)`, temperature as a grey-atmosphere scaling times `10**c`, pressure-like fields
      as straight `10**c`, radiative acceleration via `acceleration_scale * sinh(c)`.
    - `.state(labels, fields=...)` / `.log_state_jacobian(labels, fields=...)`: the subset (T, P_gas,
      m, n_e by default) and layout that `differentiable_stellar_spectroscopy`'s coupling seam
      expects; the Jacobian is `d ln(field) / d label` at fixed tau, via `jax.jacfwd`.
    - `.in_support(labels, tolerance=0.0)`: NumPy-side (non-traceable) training-box check — call it
      outside `jit`/`grad`, not inside the traced computation.
  - `LABEL_OF_FEATURE` maps Payne Zero's internal checkpoint feature names (e.g.
    `temperature_ratio_5040_k_over_temperature`) to atmojax's public label names (`teff`, `logg`, ...).
    Teff enters the network as `5040/Teff`, handled by `_features()`.
- `export.py` (`atmojax-export` entry point): loads a Payne Zero Torch checkpoint, validates its
  `format` against `SUPPORTED_FORMATS` and its coordinate field order (both are hard fail-fasts — if
  either changes upstream, the decode in `initializer.py` would silently stop matching), and writes
  everything to a plain `.npz` of float64 NumPy arrays plus per-layer `W{i}`/`b{i}` weight matrices
  (Torch `(out, in)` layout; transposed to `(in, out)` when loaded into `AtmosphereInitializer`).

### What is deliberately left out of the decode (vs. upstream Payne Zero)

Two things are intentionally not reproduced, because both have zero or undefined gradient and would
break `jax.grad`: Payne Zero's `np.clip` guards (inactive inside the training support box anyway) and
its fixed-digit text-deck rounding (a staircase function). This means atmojax's output is the
initializer's raw prediction, not a converged atmosphere — see README for the "good for / not good
for" split (good: gradient-based fits, solver tangents; not good: final uncertainties, upper-atmosphere
pressure derivatives).

### `tests/test_initializer.py`

Two tiers, controlled by whether Payne Zero + torch + its checkpoints are available:
- **Always run**: decode shape/physical-sign checks, the `labels()`/`in_support()` helper, Jacobian
  vs. central finite differences, `jit`+`grad` composition, and a weights round-trip — all against
  small synthetic weights built by `synthetic_weights()` (no real checkpoint needed).
- **Parity tests** (`test_parity_with_payne_zero`, both `five_label` and `cno8` families): compare
  atmojax's `predict()` directly against Payne Zero's own `predict()` at fixed label points. Skipped
  automatically if `payne_zero_atmosphere`/`torch` aren't importable or the checkpoint is missing/a
  Git-LFS pointer stub; force-skip with `ATMOJAX_SKIP_PARITY=1`.

### `validation/` — a separate, heavier-weight pipeline (not part of the pip package)

Answers "how good are the derivatives against a fully re-converged solver?", documented in
`validation/REPORT.md` and summarized in the README's accuracy table. Needs a working `payne-zero`
install with runtime data (line catalogs, etc.) — atmojax itself doesn't need any of this. Pipeline
order:

1. `fd_reference.py build-catalog` — one-time: memory-maps Payne Zero's sharded line catalogs into
   one combined `.npy` (needed because the stock solver loads them fully into RAM; see
   `_patch_low_memory_catalog_reads`, an experiment-local monkeypatch, not a repo-wide change).
2. `fd_reference.py solve --base {sun,arcturus,kdwarf,metalpoor_giant}` — resumable: for a base star
   and each label, re-converges Payne Zero's atmosphere solver at `label +- h` and `+- 2h` (fresh
   emulator warm start, tight tolerance), interpolates onto the canonical tau grid, and writes one
   `.npz` per perturbation under `results/runs/<base>/`.
3. `parity.py` — exports the five-label weights to `work/five_label.npz` (what `common.initializer()`
   loads) and records direct-prediction parity against Payne Zero's own `predict()`.
4. `compare.py results/runs/<base>...` — the core accuracy report: atmojax's AD Jacobian vs. two
   finite-difference references (Richardson-extrapolated `(4J(h)-J(2h))/3`, and a central-difference
   `J(h)` with its own one-sided-spread uncertainty), scored over `-3 <= log_tau <= 1` using DSS Gate
   G1/V1.3 thresholds (rel-L2 < 5%, cosine > 0.995). Writes `results/atmosphere_jacobian.json`.
5. `flux_jacobian.py results/runs/<base>` — does the atmosphere-level Jacobian translate into a
   correct *spectrum* Jacobian? Perturbs a converged base atmosphere along atmojax's AD tangent vs.
   re-synthesizing flux from fully re-converged +-h atmospheres (needs `payne_zero_synthesis`).
6. `flux_split.py results/runs/<base>` — for composition labels ([M/H], [alpha/M]) only: splits the
   flux Jacobian into a direct part (abundances change, atmosphere fixed) and an atmosphere-mediated
   part, since the direct part is shared by both arms in step 5 and flatters the emulator's score.
7. `checks.py` — supporting sanity checks quoted in the report: tau-grid agreement, solver-tolerance
   sensitivity, and label/abundance-convention parity against
   `differentiable_stellar_spectroscopy` (needs that repo checked out at `$DSS_REPO` or `../`, else
   skipped).

`validation/common.py` centralizes shared paths: `RESULTS` (checked-in `results/`, including `.npz`
run data and JSON reports — safe to read directly for numbers already computed) and `WORK`
(gitignored scratch, override with `ATMOJAX_VALIDATION_WORK`).

## Cross-repo coupling

atmojax's default `state()` field order (T, P_gas, m, n_e) and its solar-abundance/alpha-element/
helium-fraction conventions are deliberately kept identical to
[`differentiable_stellar_spectroscopy`](https://github.com/leejjoon/differentiable_stellar_spectroscopy)'s
coupling seam — don't change that ordering or those conventions without checking both sides.
