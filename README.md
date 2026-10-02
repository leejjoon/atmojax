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

[`validation/REPORT.md`](validation/REPORT.md) measures them for the five-label initializer. They are
compared against finite differences of 34 re-converged Payne Zero atmospheres (the Sun and a metal-poor
giant), at the atmosphere level and on H-band (1550–1600 nm) spectra.

| | Sun | metal-poor giant |
|---|---|---|
| ∂T/∂Teff | 0.8% | 1.5% |
| flux Jacobian, Teff / logg | 3.6% / 7.5% | 4.5% / 2.6% |
| flux Jacobian, [M/H] / [α/M] (atmosphere part only) | 1.1% / 7.0% | 3.4% / 10.6% |

**Good for:**
- driving gradient-based fits;
- serving as the tangent of a solver-valued "hybrid" seam.

**Not good for:**
- final uncertainties;
- upper-atmosphere pressure derivatives, which are off by up to about 2× above log τ ≈ −1 for Teff.

**Not yet measured:** cool giants, and the CNO initializer's derivatives. See the report for the full
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
