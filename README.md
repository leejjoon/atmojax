# atmojax

Differentiable 1D LTE model-atmosphere initializers in JAX.

`atmojax` maps stellar labels to a complete 80-layer atmosphere: column mass, temperature, gas
pressure, electron density, Rosseland opacity and radiative acceleration, on the Rosseland grid
log₁₀ τ = −6.875 + 0.125 j. It does this as a pure JAX function, so `jax.grad`, `jax.jacfwd`,
`jax.jit` and `jax.vmap` all work through it. A full label Jacobian takes about 2 ms on a CPU.

It is an independent JAX implementation of the forward pass of the atmosphere initializers released
with [Payne Zero](https://github.com/tingyuansen/payne-zero) (Ting & Kim 2026). It is not part of, or
endorsed by, the Payne Zero project. The network weights are Payne Zero's, and you export them from
your own Payne Zero installation. They are not redistributed here.

## Install

```bash
pip install -e .                 # runtime: jax + numpy only
pip install -e ".[export]"       # + torch, to export weights from a Payne Zero checkpoint
```

## Get the weights (once)

You need a Payne Zero install whose runtime data are present (its checkpoints are fetched through Git
LFS). Then:

```bash
atmojax-export five_label five_label.npz      # Teff, logg, [M/H], [alpha/M], vmic
atmojax-export cno8 cno8.npz                  # ... + [C/M], [N/M], [O/M]
# or: atmojax-export --checkpoint path/to/checkpoint.pt out.npz
```

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)     # recommended
import atmojax

init = atmojax.load("five_label.npz")
labels = init.labels(teff=5777, logg=4.44, m_h=0.0, alpha_m=0.0, vmic=1.0)

atm = init.predict_dict(labels)                # dict of (80,) arrays, CGS units
state = init.state(labels)                     # (4, 80): T, P_gas, m, n_e
J = init.log_state_jacobian(labels)            # (4, 80, 5): d ln(field) / d label at fixed tau
assert init.in_support(labels)                 # training box; check outside jit/grad
```

The default `state` layout (T, P_gas, m, n_e on the grid above) is the atmosphere state used by
[differentiable_stellar_spectroscopy](https://github.com/leejjoon/differentiable_stellar_spectroscopy)'s
coupling seam. The two codes also use identical solar abundances, α elements and helium fraction.

**What is left out:** Payne Zero's numerical clipping (inactive inside the training box) and its
fixed-digit text-deck rounding (zero derivative almost everywhere). This is the initializer's
prediction, not a converged atmosphere.

## How good are the derivatives?

[`validation/REPORT.md`](validation/REPORT.md) measures them for the five-label initializer, and for the
CNO one in the H band. They are compared against finite differences of re-converged Payne Zero
atmospheres (the Sun, a metal-poor giant and Arcturus), at the atmosphere level and on spectra.

| | Sun | metal-poor giant | Arcturus |
|---|---|---|---|
| ∂T/∂Teff | 0.8% | 1.5% | 1.9% |
| flux Jacobian, Teff / logg | 3.6% / 7.5% | 4.5% / 2.6% | 1.1% / 6.5% |
| flux Jacobian, [M/H] / [α/M] (atmosphere part only) | 1.1% / 7.0% | 3.4% / 10.6% | 1.5% / 3.6% |

**Good for:**
- driving gradient-based fits;
- serving as the tangent of a solver-valued "hybrid" seam.

**Not good for:**
- final uncertainties;
- upper-atmosphere pressure derivatives, which are off by up to about 2× above log τ ≈ −1 for Teff
  at the Sun;
- [α/M] derivatives of the atmosphere itself, worst at the cool giant (∂T/∂[α/M] 40–60% off above
  log τ ≈ −1), although little of that reaches the H-band spectrum;
- strong-line cores (Mg b, Ca II triplet), where the logg and [α/M] flux derivatives are 10–60% off;
- the CNO initializer's derivatives with respect to C, N and O, which are wrong at the atmosphere level
  (sometimes in sign) and up to 21% off in H-band flux. Its Teff, logg and [M/H] derivatives are as
  good as or better than the five-label initializer's.
  Window-averaged flux Jacobians in J, Ks and at 420–440, 510–520 and 846–870 nm are as good as the
  H band's.

**Not yet measured:** cool dwarfs, the UV, the CNO initializer outside the H band, and the
direct-abundance initializer. See the report for the full
tables, method and limitations.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The decode and derivative tests use synthetic weights and always run. The parity tests compare
against Payne Zero's own implementation for both checkpoints, and are skipped unless payne-zero,
torch and the checkpoints are available.

## License

BSD-3-Clause. The exported weights remain under Payne Zero's licence (BSD-3-Clause; see its `LICENSE`
and `NOTICE`).
