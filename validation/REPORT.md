# A differentiable Payne Zero initializer: are its label derivatives usable?

**Date:** 2026-10-01 to 2026-10-02
**Scope:** Payne Zero's five-label atmosphere initializer, ported to JAX and tested as a source of
label derivatives for [`leejjoon/differentiable_stellar_spectroscopy`](https://github.com/leejjoon/differentiable_stellar_spectroscopy) (DSS).
Every number below comes from a script in this directory and a result file under `results/`.

---

## 1. Summary

**Question.** DSS is a differentiable near-infrared synthesis pipeline in JAX. Its atmosphere is not in
the gradient graph. DSS re-converges ATLAS12 outside the graph and supplies the label Jacobian by
finite differences through a `jax.custom_jvp` seam: nine extra solves per Jacobian. DSS tried an
emulated atmosphere first (Kurucz-a1). That attempt failed DSS's Gate G1: the emulator's label
Jacobians were wrong by 3–42% even though its values were good. Can Payne Zero's fast-path
initializer, made differentiable, supply those derivatives instead?

**What was done.**
1. The five-label initializer was ported to JAX (`atmojax.AtmosphereInitializer`). It matches the original Torch/NumPy
   code to ≤ 1e-5 in every field at five test labels. A full label Jacobian costs about 2 ms on CPU.
2. Its output was checked against what DSS expects. The depth grid, the fields and the abundance
   conventions match DSS's exactly, so it can feed `dss.couple.forward.predict` without interpolation.
3. Reference label Jacobians were built from **finite differences of re-converged Payne Zero
   atmospheres**: 51 full solves at ±h and ±2h per label, at a tolerance 100× tighter than production.
   The bases were three stars: the Sun, a metal-poor giant and Arcturus (a cool giant).
4. The emulator's autodiff Jacobian was scored against the references at two levels:
   - the atmosphere itself, using DSS's G1 metrics;
   - the H-band (1550–1600 nm) **spectrum**, which is what a fit consumes.

**Answer.** The emulator's label derivatives are good enough to drive a fit, and much better than the
emulator DSS rejected. They are not good enough to use directly as uncertainties.

| | Sun | metal-poor giant | Arcturus |
|---|---|---|---|
| ∂T/∂Teff (the cell DSS's emulator failed by 2.6–14.5%) | **0.8%** | **1.5%** | **1.9%** |
| flux Jacobian, Teff | 3.6% | 4.5% | 1.1% |
| flux Jacobian, logg | 7.5% | 2.6% | 6.5% |
| flux Jacobian, [M/H] (atmosphere part only) | 1.1% | 3.4% | 1.5% |
| flux Jacobian, [α/M] (atmosphere part only) | 7.0% | 10.6% | 3.6% |
| cosine of every flux Jacobian | ≥ 0.998 | ≥ 0.997 | ≥ 0.998 |

The emulator fails in two specific places. At the Sun, the gas-pressure and column-mass derivatives
are wrong in the upper atmosphere (up to about 2× above log τ ≈ −1 for Teff); at the giants this is
much milder. The [α/M] column is 6–28% off at every base, and worst at Arcturus, where ∂T/∂[α/M] is
40–60% off above log τ = −1. This is exactly the error mechanism DSS documented, appearing exactly
where its arithmetic predicts. Most of it does not reach the H-band spectrum, which forms deeper, nor
the window-averaged flux Jacobian in the J and Ks bands or at 420–440, 510–520 or 846–870 nm.
Strong-line cores are the exception: in the Mg b and Ca II triplet cores the logg and [α/M] flux derivatives are 10–60% off
(§6.2).

**Recommendation for DSS.** Use a hybrid seam:
- **Value:** take the atmosphere from a converged solve, as DSS does now.
- **Tangent during the fit:** take it from the emulator's autodiff.
- **At convergence:** recompute the finite-difference seam Jacobian once, for the uncertainties.

A Gauss–Newton step with a Jacobian 2–10% off still converges to the same optimum, because the
optimum is set by exact residuals. So this removes most of the solver calls without biasing the
result.

**The cool giant does not change this.** Cool stars are where DSS found the worst emulator errors.
At Arcturus (4286 K) the atmosphere-level [α/M] Jacobian is the worst of the three bases, but the
H-band flux Jacobians are as good as or better than at the other two (1.1–6.5%, cosine ≥ 0.998).
This is one cool giant, not the cool-star regime (§8).

**Against ATLAS12** (§5.5), using DSS's own stored V1.3 solves, atmojax scores as it does against
Payne Zero (the reference's bias is small). It beats the emulator DSS rejected in 59 of 65 cells
(median error 4.6% against 10.4%), though it still fails G1 on most cells.

**The CNO initializer** (§6.3) is as good as or better than the five-label one for Teff, logg and
[M/H], but its derivatives with respect to C, N and O are not usable: several atmosphere cells have the
wrong sign. In the H band most of the C, N and O signal is direct line opacity, so the flux Jacobian is
still within 1–4% where the atmosphere's share is small, but 11–21% off where it is not.

---

## 2. Background

### 2.1 Payne Zero's fast path, and why it has no label gradients

`payne_zero_synthesis.synthesize_from_labels` predicts an atmosphere with the initializer, rebuilds
populations, and synthesizes. It never runs the iterative solver. The chain from labels to flux leaves
the autograd graph at four places:

1. **Initializer** (`payne_zero_atmosphere/warm_start.py`):
   - The network runs under `torch.no_grad()`.
   - The PCA decode is done in NumPy.
   - The prediction is written out as a fixed-digit text deck and parsed back. That rounding is
     load-bearing for the certified solver, but it has zero derivative almost everywhere.
2. **Population bridge** (`payne_zero_synthesis/pipeline.py:308`): host-fp64 equation of state, with
   detached iterative solves.
3. **Pipeline ingestion** (`pipeline.py:1109`): `np.asarray(..., float64)` on every column.
4. **Host-side table lookups** inside the continuum opacity (for example `continuum.py:1971`).

Synthesis is differentiable with respect to line parameters at a fixed atmosphere. That is what
`linelist_calibration` uses. It is not differentiable with respect to anything that moves the
atmosphere.

### 2.2 DSS and Gate G1

DSS (`plans/phase1_failure_analysis.md`) tested the Kurucz-a1 atmosphere emulator.
- **Values:** median errors of 0.4–7%.
- **Label Jacobians:** wrong by 3–42%. It failed 17 of 19 held-out cells at the G1 threshold
  (rel-L2 < 5%, cosine > 0.995).
- **Fixes:** Sobolev retraining improved only the trained points and made held-out points worse.

DSS's diagnosis is general. A value-trained network's error is a smooth field of a few millidex
that varies over hundreds of kelvin, and the true slopes are small (about 9×10⁻⁵ dex/K in log T).
So the error field's tilt competes with the true derivative, and a good value fit puts no bound on
the derivative. DSS therefore keeps the atmosphere out of the graph. `dss/couple/atmosphere.py:486`
(`AtlasResolver.state_fn`) maps labels (Teff, logg, [M/H], [α/M]) to a (4, 80) array of T, P_gas, m,
n_e on log τ_Ross = −6.875 + 0.125j. The value comes from ATLAS12 and the tangent from finite
differences of re-converged models.

Payne Zero's initializer has never been tested this way. Payne Zero itself only uses it as a warm
start.

---

## 3. The JAX initializer (`src/atmojax/initializer.py`)

**The checkpoint** (`source_data_files/atmosphere_emulator/five_label/checkpoint.pt`, format
`payne_zero_complete_atmosphere_latent_v2`):
- **Inputs:** 5 features (5040/Teff, logg, [M/H], [α/M], ξ), standardized.
- **Network:** an MLP of 6 hidden layers × 1024 with SiLU activations, outputting 160 PCA coefficients.
- **Decode:** a 160 × 480 PCA basis, giving 80 layers × 6 coordinate fields.
- **Training:**
  - 50,000 converged Payne Zero atmospheres, plus 2,000 for early stopping.
  - Its loss term `derivative_weight` = 0.1 matches profile slopes *in depth*, not label derivatives.
  - Eight named reference stars were excluded from training (`fixed_gate_slugs_excluded`), including
    `sun` and `giant`.

**The port** reproduces `AtmosphereInitializer.predict`:
- standardize the inputs;
- run the MLP in float32, as the original does;
- de-standardize the PCA coefficients and expand them;
- decode the six fields:
  - column mass = cumsum of 10^c₀ (always increasing);
  - T = T_grey(τ) · 10^c₁;
  - P_gas, n_e, κ_Ross = 10^c;
  - g_rad = scale · sinh(c₅).

Two things are deliberately left out. The `np.clip` guards are inactive inside the training support.
The text-deck round trip is a staircase function.

**Parity with the original code** (`results/parity.json`): the maximum relative difference per field
at five labels is ≤ 6.5e-6 for column mass, T, P_gas, n_e and κ_Ross. For the signed g_rad,
normalized by its maximum, it is ≤ 4.2e-5. This is float32 rounding in the MLP.

**Cost:** `jax.jacfwd` of ln(T, P_gas, m, n_e) with respect to 5 labels takes about 1–2 ms after
JIT compilation, on CPU.

**The eight-label CNO checkpoint** has the same architecture with 8 inputs. atmojax supports it, and
`tests/test_initializer.py` checks its parity with Payne Zero's own code. §4–§6.2 concern the
five-label initializer; §6.3 validates the CNO one in the H band.

**Compatibility with DSS** (`results/checks.json`):
- **Depth grid:** the checkpoint's `standard_rosseland_optical_depth` is exactly
  10^(−6.875 + 0.125j), j = 0…79, which is DSS's canonical grid.
- **Fields:** `AtmosphereInitializer.state` returns DSS's (4, 80) layout of T, P_gas, m, n_e by default.
- **Abundance conventions:** both codes use identical solar log abundances for Z = 3–99 (maximum
  difference 0.0), the same α set (O, Ne, Mg, Si, S, Ca, Ti), and the same helium fraction (0.078370).
- **Microturbulence:** DSS's seam has no microturbulence label, so ξ must be pinned to the value DSS
  uses.

---

## 4. Reference Jacobians (`fd_reference.py`)

### 4.1 Construction

**Bases.**
- **Sun:** Teff 5777 K, logg 4.44, [M/H] 0, [α/M] 0, ξ 1.0 km/s.
- **Metal-poor giant:** 5000 K, logg 2.50, [M/H] −1.50, [α/M] +0.40, ξ 1.5 km/s. This is an
  in-support point that I chose, not a named excluded star, so it is not verified to be held out.
- **Arcturus:** 4286 K, logg 1.66, [M/H] −0.52, [α/M] +0.30, ξ 1.7 km/s. It is not one of the named
  excluded stars either; the nearest one, `giant`, is at 4500 K, logg 2.0, [M/H] −0.5, [α/M] +0.2.
  Its first attempt ran out of memory; it was solved later on a larger machine (§4.3).

**Perturbations.** Each of Teff, logg, [M/H] and [α/M] was perturbed by ±h and ±2h, with
h = 25 K, 0.05, 0.05 and 0.05 respectively. The same steps as DSS's V1.3. Abundances are stored to
0.01 dex, so the composition steps are exact. That gives 17 solves per base.

**Solver settings.**
- Production physics: molecules, convection and the full source-line catalogs.
- Each solve is started fresh from the emulator at the perturbed labels.
- Convergence stop: maximum deep-layer relative temperature change below 5×10⁻⁶ on two consecutive
  iterations. The all-layer change must be below 5×10⁻⁵, with at least 5 iterations and at most 60.
  Production uses 5×10⁻⁴ with one consecutive iteration.
- The state is taken in float64 from `result.atmosphere`, never through the text deck.

**Depth registration.** T, P_gas, m and n_e are interpolated with PCHIP in log₁₀ onto the canonical
grid, using each model's own integrated Rosseland depth, as DSS does. The converged models are already
on that grid to a median of 1.0×10⁻⁴ dex (Sun) and 1.5×10⁻⁴ dex (giant). The maximum is about
0.01 dex at the surface.

**References.** Two per cell:
- **Richardson** (4J(h) − J(2h))/3, scored only on layers where |J(h) − J(2h)|/|J| < 5%. This is
  DSS's V1.3 construction.
- **Central ±h**, on every layer, reported with its own uncertainty: the disagreement between the
  forward and backward one-sided differences. An emulator error no larger than that spread is inside
  the reference's noise. It cannot be called a failure.

### 4.2 Convergence

| | solves | converged at 5e-6 | stopped at the cap | mean iterations | mean wall time |
|---|---|---|---|---|---|
| Sun | 17 | 15 | 2: Teff+2h at 1.3e-4, [α/M]+2h at 1.7e-5 | 31.6 | 285 s |
| giant | 17 | 9 | 8: six of them at 6e-6 to 1.3e-5; Teff+2h at 1.1e-4 and [M/H]−2h at 2.8e-4 | 55.1 | 363 s |
| Arcturus | 17 | 13 | 4: [M/H]+h at 5.0e-6, [M/H]+2h at 1.1e-5, [α/M]+2h at 1.3e-5; [α/M]−2h at 7.7e-4 | 37.4 | 635 s |

**Teff + 50 K never converged at the Sun or the giant.** Its per-iteration change plateaued around
1e-4. DSS saw the same limit-cycling with ATLAS12. At Arcturus it converged in 24 iterations. The
affected Richardson cells lose layers to the stability filter. The central ±h reference does not use
those points. At Arcturus the unusable point is instead [α/M] − 2h (7.7e-4), so only the Arcturus
[α/M] Richardson column is affected; its central ±h column agrees with it (§5.3).

**The tolerance is not a limiting error** (`results/checks.json`). Re-solving the Sun base at 2e-5
instead of 5e-6 moves ln T by a median of 2.1×10⁻⁵ over the band. That is 0.5% of the 25 K Teff
signal (4.2×10⁻³).

Wall times are for 4 CPU threads, at roughly 10–15 s per iteration (Sun, giant) and about 17 s per
iteration (Arcturus). The Arcturus solves ran on a different machine (Xeon Gold 6526Y), four at a
time, so their wall times are not strictly comparable.

### 4.3 Running in a 15 GB container

Stock Payne Zero keeps every source catalog resident in RAM. The atmosphere opacity uses about 7 GB of
line lists, and `np.concatenate` over the three 1.4 GB predicted-line shards peaks near 8.4 GB on its
own. The out-of-memory killer stopped the first two Sun attempts at about 13.9 GB.

**The fix** (`validation/fd_reference.py`; validation-local, no Payne Zero code was changed):
- `build-catalog` writes the three shards, in order, into one 4.2 GB `.npy`.
- The standard and diatomic catalog readers are patched to return memory maps (`mmap_mode="r"`):
  identical bytes, held as file-backed pages.
- **Result:** the Sun solve ran at about 8.9 GB of process memory plus 4.7 GB of reclaimable file
  pages (`results/logs/mem.log`).

**Arcturus** still ran out of memory in iteration 3, at 13.9 GB of process memory. In a cool giant
many more molecular and weak atomic lines pass the opacity selection, so the per-iteration working
arrays grow. It needed a machine with more memory, not a code change.

**Arcturus on a 128 GB machine.** With the same patch and 4 threads, the base solve peaked at 19.9 GB
maximum resident set size, which includes the memory-mapped catalog pages; each of the 16 perturbation
solves peaked at about 19.5 GB. Running four solves at once used at most 71 GB of the machine in total,
and the 17 solves finished in 61 minutes (`results/logs/arcturus_*.log`). A single Arcturus solve
therefore needs a machine with about 24 GB or more, not the 32 GB first estimated.

---

## 5. Atmosphere-level results (`compare.py`, `results/atmosphere_jacobian.json`)

**Metrics:** rel-L2 = ‖J_emu − J_ref‖/‖J_ref‖ and cosine, both over −3 ≤ log τ ≤ 1, on
d ln(field)/d label. **G1 pass:** rel-L2 < 5% and cosine > 0.995. The "spread" column is the
reference's own one-sided uncertainty (§4.1).

**Value errors of the emulator at the base** (median |Δ ln| over the band):

| | T | P_gas | m | n_e |
|---|---|---|---|---|
| Sun | 0.06% | 2.3% | 0.23% | 0.95% |
| giant | 0.06% | 3.3% | 0.41% | 1.1% |
| Arcturus | 0.05% | 1.2% | 3.2% | 1.5% |

### 5.1 Sun

| cell | Richardson rel / cos / layers | central ±h rel | ref. spread | verdict |
|---|---|---|---|---|
| T / Teff | 0.009 / 1.0000 / 33 | 0.008 | 0.020 | **pass** |
| P_gas / Teff | 0.089 / 0.9960 / 27 | 0.110 | 0.048 | **fail** |
| m / Teff | 0.143 / 0.9920 / 33 | 0.142 | 0.034 | **fail** |
| n_e / Teff | 0.054 / 0.9986 / 33 | 0.053 | 0.046 | marginal |
| T / logg | 0.064 / 0.9980 / 13 | 0.145 | 0.200 | inconclusive |
| P_gas / logg | 0.027 / 1.0000 / 33 | 0.027 | 0.005 | pass |
| m / logg | 0.018 / 1.0000 / 33 | 0.018 | 0.001 | pass |
| n_e / logg | 0.017 / 0.9999 / 33 | 0.015 | 0.017 | pass |
| T / [M/H] | 0.051 / 0.9988 / 27 | 0.102 | 0.115 | inconclusive |
| P_gas / [M/H] | 0.020 / 0.9998 / 33 | 0.019 | 0.014 | pass |
| m / [M/H] | 0.024 / 1.0000 / 33 | 0.024 | 0.012 | pass |
| n_e / [M/H] | 0.032 / 0.9996 / 28 | 0.039 | 0.045 | pass |
| T / [α/M] | 0.159 / 0.9877 / 21 | 0.159 | 0.169 | inconclusive |
| P_gas / [α/M] | 0.078 / 0.9970 / 33 | 0.078 | 0.047 | fail (mild) |
| m / [α/M] | 0.082 / 0.9975 / 33 | 0.081 | 0.047 | fail (mild) |
| n_e / [α/M] | 0.077 / 0.9976 / 25 | 0.087 | 0.064 | marginal |

### 5.2 Metal-poor giant

| cell | Richardson rel / cos / layers | central ±h rel | ref. spread | verdict |
|---|---|---|---|---|
| T / Teff | 0.017 / 0.9998 / 33 | 0.015 | 0.036 | **pass** |
| P_gas / Teff | 0.048 / 0.9998 / 33 | 0.046 | 0.039 | pass |
| m / Teff | 0.084 / 0.9994 / 33 | 0.084 | 0.034 | fail |
| n_e / Teff | 0.037 / 0.9995 / 33 | 0.032 | 0.072 | pass |
| T / logg | 0.059 / 0.9984 / 18 | 0.096 | 0.128 | inconclusive |
| P_gas / logg | 0.011 / 1.0000 / 33 | 0.011 | 0.005 | pass |
| m / logg | 0.016 / 1.0000 / 33 | 0.016 | 0.002 | pass |
| n_e / logg | 0.018 / 0.9998 / 33 | 0.016 | 0.021 | pass |
| T / [M/H] | 0.059 / 0.9983 / 22 | 0.090 | 0.079 | inconclusive |
| P_gas / [M/H] | 0.010 / 1.0000 / 33 | 0.010 | 0.022 | pass |
| m / [M/H] | 0.029 / 1.0000 / 33 | 0.029 | 0.024 | pass |
| n_e / [M/H] | 0.046 / 0.9997 / 25 | 0.052 | 0.033 | pass |
| T / [α/M] | 0.102 / 0.9963 / 25 | 0.163 | 0.493 | inconclusive |
| P_gas / [α/M] | 0.205 / 0.9994 / 33 | 0.205 | 0.040 | **fail** |
| m / [α/M] | 0.088 / 0.9990 / 33 | 0.088 | 0.040 | fail |
| n_e / [α/M] | 0.081 / 0.9994 / 24 | 0.121 | 0.160 | inconclusive |

### 5.3 Arcturus

| cell | Richardson rel / cos / layers | central ±h rel | ref. spread | verdict |
|---|---|---|---|---|
| T / Teff | 0.019 / 0.9998 / 33 | 0.019 | 0.009 | **pass** |
| P_gas / Teff | 0.035 / 0.9997 / 33 | 0.034 | 0.033 | pass |
| m / Teff | 0.072 / 0.9998 / 33 | 0.073 | 0.031 | fail (mild) |
| n_e / Teff | 0.057 / 0.9986 / 33 | 0.056 | 0.042 | marginal |
| T / logg | 0.174 / 0.9987 / 13 | 0.394 | 0.297 | inconclusive |
| P_gas / logg | 0.015 / 1.0000 / 33 | 0.015 | 0.006 | pass |
| m / logg | 0.030 / 1.0000 / 33 | 0.030 | 0.002 | pass |
| n_e / logg | 0.030 / 0.9997 / 33 | 0.030 | 0.014 | pass |
| T / [M/H] | 0.100 / 0.9989 / 15 | 0.216 | 0.157 | inconclusive |
| P_gas / [M/H] | 0.020 / 0.9999 / 33 | 0.020 | 0.012 | pass |
| m / [M/H] | 0.015 / 0.9999 / 33 | 0.015 | 0.005 | pass |
| n_e / [M/H] | 0.042 / 0.9995 / 27 | 0.053 | 0.033 | pass |
| T / [α/M] | 0.278 / 0.9643 / 16 | 0.336 | 0.130 | **fail** |
| P_gas / [α/M] | 0.057 / 0.9991 / 33 | 0.057 | 0.037 | marginal |
| m / [α/M] | 0.113 / 0.9985 / 33 | 0.113 | 0.031 | fail |
| n_e / [α/M] | 0.120 / 0.9955 / 30 | 0.123 | 0.032 | fail |

The [α/M] Richardson column uses the poorly converged [α/M] − 2h solve (§4.2). The central ±h column,
which does not, gives the same verdicts.

### 5.4 Reading the tables

- **Temperature against Teff passes at all three bases** (0.8%, 1.5%, 1.9%), within or close to the
  reference's uncertainty. That is the most important cell for line formation. It is also the one
  where DSS's emulator failed: −14.5% at τ = 1 at the Sun, with a band median of −2.6%, and 17 of 19 held-out
  cells failing overall.

- **Pressure-like fields against logg and [M/H] pass** at all three bases (1–5%).

- **Temperature against logg and [M/H] cannot be judged here, nor T against [α/M] at the Sun and the
  giant.** T barely responds to these labels, and the forward and backward differences disagree by
  8–49%, as much as or more than the emulator's apparent error. Deciding these cells would need larger
  steps or a smoother reference. Arcturus's T/[α/M] is the exception (below).

- **Pressure and column mass against Teff fail at the Sun, and the failure is localized in depth.**
  `rel_l2_by_log_tau` in the JSON gives:

  | log τ range | P_gas / Teff error | m / Teff error |
  |---|---|---|
  | −5 to −3 | 78% | 79% |
  | −3 to −1 | 93% | 120% |
  | −1 to 1 | 7% | 10% |
  | 1 to 2 | 1.6% | 1.9% |

  In the upper layers the emulator's d ln P/d Teff is about −8×10⁻⁵ K⁻¹ against a true −3.5×10⁻⁵,
  roughly 2× too steep. Below log τ ≈ 0 the two agree to about 2% (for example log τ = 0.875:
  −6.24×10⁻⁵ against −6.39×10⁻⁵). This is DSS's mechanism exactly. The emulator's pressure value
  error is about 2–3%, while the true ∂ln P/∂Teff in the upper atmosphere is only about 0.035 per
  1000 K. Any tilt in a 2% error field across a few hundred kelvin is comparable to the signal.
  In the giant the same pattern is far milder. P_gas/Teff is off by 10.5% (log τ −5 to −3),
  7.6% (−3 to −1) and 4.2% (−1 to 1); m/Teff peaks at 17% (−3 to −1). So the upper-atmosphere
  failure varies from star to star in size, which is also what DSS's mechanism predicts.
  At Arcturus it is absent: P_gas/Teff is at most 5.3% in every depth range, and m/Teff peaks at 10%
  (−1 to 1).

- **The [α/M] column is the weakest, and worst at Arcturus.** At the Sun and the giant the worst case
  is the giant's P_gas/[α/M] at 20%. At Arcturus T/[α/M] fails clearly, the only temperature cell
  at any base that does: 34% against a reference spread of 13%. By depth it is 40% (log τ −5 to −3),
  62% (−3 to −1), 26% (−1 to 1) and 4.7% (1 to 2). m/[α/M] and n_e/[α/M] are off by 11–12% against a
  spread of 3%. Cool, molecule-rich atmospheres are where DSS's emulator was worst too. §6 shows how
  little of this reaches the H band.

### 5.5 Against ATLAS12 (`atlas12_compare.py`, `results/atlas12_jacobian.json`)

The references above come from Payne Zero's solver, which the emulator was trained to imitate. DSS's
Phase 1 V1.3 campaign stored ATLAS12 (pyKurucz) solves at ±h, ±2h and ±4h in the four labels around
12 bases, with the same steps (`artifacts/phase1/fd/` in DSS). `atlas12_compare.py` reads those decks,
registers them on the canonical τ grid as `fd_reference.py` does, and scores atmojax with the same
metrics; no solves are run. Five bases lie inside the initializer's training box: the Sun and Arcturus
(DSS's test split) and three of DSS's training points. The other seven are at the grid's corners
(3800 K or 7500 K) and are skipped. All were solved at ξ = 2 km/s, and less tightly than §4: the final
per-iteration change is 5×10⁻⁶ to 10⁻³, which shows up as larger reference spreads.

| base | G1 passes vs ATLAS12 | vs Payne Zero (§5) | cells clearly failing (error > 2× spread and > 10%) |
|---|---|---|---|
| Sun | 7/16 | 7/16 | P_gas/[α/M] 16%, m/Teff 15%, P_gas/Teff 11% |
| Arcturus | 8/16 | 8/16 | n_e/[α/M] 14%, m/[α/M] 12% |
| 5463 K, logg 3.51, [M/H] −2.10, [α/M] +0.16 | 7/16 | – | P_gas/[α/M] 46%, m/[α/M] 39% |
| 4462 K, logg 3.63, [M/H] −0.73, [α/M] +0.22 | 9/16 | – | P_gas/[α/M] 13%, P_gas/Teff 11% |
| 5026 K, logg 1.50, [M/H] −2.33, [α/M] +0.31 | 4/16 | – | m/[α/M] 45% |

**The reference's bias is small.** At the Sun and Arcturus the scores against ATLAS12 are close to those
against Payne Zero, cell by cell, with the same pass counts. ∂T/∂Teff is 2.5% and 1.9% off. The
exception is [α/M] at the Sun (P_gas/[α/M] 16% against ATLAS12, 8% against Payne Zero). The weak spots
are the ones already found: [α/M] everywhere, worst at [M/H] ≈ −2, and [M/H] at [M/H] ≈ −2 (7–22%).

**Head to head with DSS's emulator.** The same ATLAS12 solves are what DSS used to reject its Kurucz-a1
emulator (`fd/v1_3_report.json`). Rescoring atmojax with that report's band (−4 ≤ log τ ≤ 1) and
stricter 1% stability threshold, on the 65 cells both scored at these five bases:

| | atmojax | DSS's Kurucz-a1 emulator |
|---|---|---|
| better in | **59 of 65 cells** | 6 |
| median rel-L2 | **4.6%** | 10.4% |
| worst cell | 54% | 231% |
| G1 passes (of 80) | **35** | 13 |

atmojax's tangent is better than the emulator DSS measured at every base, by more than 2× in median at
four of the five. It still fails G1 on most cells.

---

## 6. Spectrum-level results (`flux_jacobian.py`, `flux_split.py`)

An atmosphere-level failure matters only insofar as it changes the spectrum's derivative, which is
what a fit uses. For each label, two central differences of normalized flux were computed with
**identical abundances on both arms**:

- **reference:** Payne Zero synthesis of the atmospheres re-converged at l ± h;
- **emulator:** the converged base atmosphere moved along the emulator's autodiff tangent, ±h in each
  field (ln T, ln P_gas, ln m, ln n_e). This is what the hybrid seam would supply.

A third arm moves only T along the emulator tangent and takes P_gas, m and n_e from the reference.
It attributes the error to temperature or to the pressure-like fields.

The window is 1550–1600 nm, the H band where DSS and Payne Zero calibrate, with R_grid = 100,000
(3,175 pixels). Molecular lines are on, everything is in float64, and synthesis runs on CPU.

**Flux-Jacobian error** (`results/runs/*/flux_jacobian.json`):

| base | label | rel-L2 | cos | gain | T-only arm |
|---|---|---|---|---|---|
| Sun | Teff | 0.036 | 0.9999 | 1.032 | 0.005 |
| Sun | logg | 0.075 | 0.9979 | 1.034 | 0.013 |
| Sun | [M/H] | 0.006 | 1.0000 | 1.001 | 0.002 |
| Sun | [α/M] | 0.051 | 0.9987 | 1.007 | 0.011 |
| giant | Teff | 0.045 | 0.9991 | 1.015 | 0.012 |
| giant | logg | 0.026 | 0.9999 | 0.979 | 0.031 |
| giant | [M/H] | 0.013 | 1.0000 | 0.988 | 0.008 |
| giant | [α/M] | 0.053 | 0.9986 | 0.990 | 0.009 |
| Arcturus | Teff | 0.011 | 0.9999 | 1.001 | 0.008 |
| Arcturus | logg | 0.065 | 0.9983 | 1.025 | 0.008 |
| Arcturus | [M/H] | 0.013 | 1.0000 | 0.989 | 0.003 |
| Arcturus | [α/M] | 0.032 | 0.9995 | 1.011 | 0.008 |

"Gain" is the projection of the emulator's Jacobian onto the reference's: ⟨J_emu, J_ref⟩ / ‖J_ref‖².
A gain of 1.03 means the fit would see that label's sensitivity 3% too large.

**[M/H] and [α/M], scored on the atmosphere part only** (`results/runs/*/flux_split.json`). The direct
abundance term (base atmosphere held fixed, abundances at l ± h) is common to both arms, so it flatters
the scores above. Removing it:

| base | label | atmosphere share of the total | emulator error on it |
|---|---|---|---|
| Sun | [M/H] | 59% | 1.1% (cos 0.9999) |
| Sun | [α/M] | 73% | 7.0% (cos 0.9976) |
| giant | [M/H] | 38% | 3.4% (cos 0.9999) |
| giant | [α/M] | 50% | 10.6% (cos 0.9966) |
| Arcturus | [M/H] | 86% | 1.5% (cos 0.9999) |
| Arcturus | [α/M] | 89% | 3.6% (cos 0.9996) |

**Reading.**
- **Every flux Jacobian points the right way** (cosine ≥ 0.997). Magnitudes are within 1–8% for
  Teff, logg and [M/H], and within 4–11% for the atmosphere part of [α/M].
- **The upper-atmosphere pressure failure barely reaches the H band.** With the emulator's temperature
  derivative and the reference pressure derivatives, every label at every base is at 0.2–3.1%. So the residual flux
  error comes from the pressure-like fields, but is much smaller than their atmosphere-level error,
  because the H-band continuum and most lines form deeper.
- **Exception: giant logg.** There the T-only arm (3.1%) is worse than the full emulator (2.6%), so the
  T and pressure errors partly cancel.
- **Arcturus's [α/M] failure barely reaches the H band either.** It has the worst atmosphere-level
  [α/M] Jacobian (§5.4) but the best flux score for the atmosphere part of [α/M] (3.6%). That part is
  89% of the total at Arcturus, so the direct abundance term is not what makes the score look good.
  With the emulator's temperature tangent alone the error is 0.8%: T's absolute response to [α/M] is
  small, so even a 26% error in it near log τ = 0 moves the flux little. The rest comes from
  m and n_e.
- **Wavelength dependence.** Bands whose lines form higher (strong-line cores, the blue and UV) were
  expected to be more sensitive to the upper-atmosphere pressure error. §6.1 measures three more
  windows; §6.2 measures the strong-line cores themselves.

### 6.1 Other bands

The same test, with the same 51 solves, in five more windows. Three are where more of the spectrum
forms higher than in the H band: 420–440 nm (blue, with the CH G band; 4,652 pixels), 510–520 nm
(Mg I b; 1,942 pixels) and 846–870 nm (the Gaia RVS window, with the Ca II triplet; 2,797 pixels).
Two are the other near-infrared bands: J at 1170–1330 nm (12,818 pixels) and Ks at 2000–2300 nm,
including the CO bandheads (13,976 pixels). All at R_grid = 100,000
(`results/runs/*/flux_jacobian_<band>nm.json`, `flux_split_<band>nm.json`).

**Flux-Jacobian rel-L2** (cosine in parentheses where it is below 0.998; [M/H] and [α/M] on the
atmosphere part only):

| base | band (nm) | Teff | logg | [M/H] atm. | [α/M] atm. |
|---|---|---|---|---|---|
| Sun | 1550–1600 | 0.036 | 0.075 (0.9979) | 0.011 | 0.070 (0.9976) |
| Sun | 420–440 | 0.037 | 0.110 (0.9966) | 0.007 | 0.054 |
| Sun | 510–520 | 0.037 | 0.073 | 0.008 | 0.069 |
| Sun | 846–870 | 0.067 | 0.117 (0.9940) | 0.017 | 0.079 |
| Sun | 1170–1330 (J) | 0.037 | 0.073 | 0.012 | 0.089 (0.9971) |
| Sun | 2000–2300 (Ks) | 0.034 | 0.065 | 0.012 | 0.081 (0.9973) |
| giant | 1550–1600 | 0.045 | 0.026 | 0.034 | 0.106 (0.9966) |
| giant | 420–440 | 0.023 | 0.022 | 0.030 | 0.078 |
| giant | 510–520 | 0.015 | 0.017 | 0.030 | 0.087 |
| giant | 846–870 | 0.028 | 0.037 | 0.038 | 0.048 |
| giant | 1170–1330 (J) | 0.019 | 0.030 | 0.036 | 0.060 |
| giant | 2000–2300 (Ks) | 0.018 | 0.057 | 0.034 | 0.055 |
| Arcturus | 1550–1600 | 0.011 | 0.065 | 0.015 | 0.036 |
| Arcturus | 420–440 | 0.009 | 0.076 (0.9974) | 0.015 | 0.038 |
| Arcturus | 510–520 | 0.008 | 0.069 (0.9979) | 0.016 | 0.045 |
| Arcturus | 846–870 | 0.012 | 0.055 | 0.025 | 0.080 |
| Arcturus | 1170–1330 (J) | 0.013 | 0.063 | 0.025 | 0.065 |
| Arcturus | 2000–2300 (Ks) | 0.013 | 0.054 | 0.017 | 0.052 |

**Window-averaged, no band is much worse than the H band.** Every cosine is ≥ 0.994, Teff and [M/H] stay
below 7%, and the atmosphere part of [α/M] within 4–11%. logg is the weakest label, worst at the Sun
in the blue and in the RVS window (11–12%, gain 1.03–1.07). In the Sun's blue window the atmosphere
part of [M/H] and [α/M] is larger than the total (share 1.13 and 1.09): the direct and atmosphere
terms partly cancel there. J and Ks behave like the H band: Teff 1–4%, logg 3–7%, [M/H] 1–4%, the
atmosphere part of [α/M] 5–9%, every cosine ≥ 0.997. The near-infrared result does not depend on which
of J, H or Ks a fit uses.

### 6.2 Strong-line cores (`flux_cores.py`)

A window-wide rel-L2 is dominated by weak lines and wings, so it hides what happens where the
spectrum forms highest. `flux_cores.py` synthesizes the base spectrum, finds the minimum of each
strong line, takes the contiguous pixels whose depth is at least 80% of the minimum's, and scores the
saved flux-Jacobian arms on those pixels only (9–20 pixels per line group). A core responds weakly
and holds few pixels, so the reference's own noise is reported as in §5: the disagreement of the two
one-sided differences (`results/runs/*/flux_cores.json`).

**rel-L2 of the emulator's flux Jacobian in the cores / reference spread** (bold: error well above
the spread):

| base | lines | Teff | logg | [M/H] | [α/M] |
|---|---|---|---|---|---|
| Sun | Ca II triplet | 0.113 / 0.077 | **0.219** / 0.010 | **0.097** / 0.013 | **0.092** / 0.018 |
| Sun | Mg I b | 0.027 / 0.015 | **0.622** / 0.028 | 0.046 / 0.031 | **0.149** / 0.026 |
| giant | Ca II triplet | 0.062 / 0.066 | 0.044 / 0.023 | 0.033 / 0.020 | **0.198** / 0.043 |
| giant | Mg I b | 0.009 / 0.019 | 0.539 / 0.324 | 0.019 / 0.028 | 0.073 / 0.039 |
| Arcturus | Ca II triplet | 0.171 / 0.351 | 0.034 / 0.022 | 0.006 / 0.054 | **0.284** / 0.061 |
| Arcturus | Mg I b | 0.041 / 0.080 | **0.104** / 0.018 | 0.007 / 0.036 | **0.157** / 0.038 |

**Reading.**
- **Strong-line cores expose errors of 10–60% that the window averages hide.** The worst is the Sun's
  Mg b logg derivative: 62% off, cosine 0.84, against a reference spread of 3%.
- **For logg and [M/H] the error is the temperature tangent.** The arm with only the emulator's T
  reproduces it: 59% of the Sun's Mg b logg error, 19% of its Ca II logg error, 9% of its Ca II [M/H]
  error. These are exactly the T/logg and T/[M/H] cells that §5 could not resolve, because T barely
  responds there and the atmosphere-level reference was too noisy. Line cores are a more sensitive
  probe of them, and in the upper atmosphere the emulator's ∂T/∂logg is wrong.
- **For [α/M] it is mixed.** At the Sun and Arcturus the T-only arm is 3–14%, so most of the core error
  comes from the pressure-like fields. At the giant's Ca II cores it comes from T (22%).
- **J and Ks have no comparable cores.** `flux_cores.py` also scores the K I doublet (1243, 1252 nm)
  and the Na I doublet (2206, 2208 nm), but at this sampling their cores are one pixel per line, and
  the reference spread on two pixels is up to 42%. Those entries in `flux_cores.json` are not
  informative and are left out of the table.
- **Caveat: these are LTE cores.** Real Ca II triplet and Mg b cores carry NLTE and chromospheric
  effects that no 1D LTE model reproduces, so a fit would usually down-weight or mask them anyway.
- **Consequence.** A fit across a whole window is still driven by the window-averaged Jacobian
  (§6.1). A fit dominated by strong-line cores, for example a logg estimate from Mg b alone, would see
  sensitivities tens of percent off. That is a step-size problem for the optimizer, not a bias, but it
  is a reason to keep the finite-difference Jacobian for uncertainties.

### 6.3 The CNO (eight-label) initializer

Payne Zero's second initializer adds [C/M], [N/M] and [O/M] as labels (atmojax family `cno8`). It was
scored the same way, at the same three bases, with C = N = 0 and [O/M] = [α/M] at the base.

**Reference solves** (`fd_reference.py --family cno8`, `results/runs_cno8/`).
- **Reused:** that base mixture is exactly the five-label one (Payne Zero's five-label path treats O as
  an α element; the two warm starts' abundance tables are identical at every base), so the five-label
  Teff, logg and [M/H] solves serve as references here too.
- **New:** C, N and O at ±h and ±2h (h = 0.05 dex), and [α/M] at ±h, ±2h **with [O/M] held fixed**: in
  this family O is its own label, so the five-label α solves, which move O as well, do not apply.
- **Warm-start check:** each base was re-solved from the CNO initializer's warm start. Over the band it
  agrees with the five-label base to a median of 1e-7 to 2e-5 in ln of every field (maximum 4e-3 in
  n_e at a few layers), the same level as the tolerance check in §4.2.
- **Convergence:** 51 solves; 38 converged at 5×10⁻⁶ and 13 stopped at the cap between 4.4×10⁻⁶ and
  3.9×10⁻⁵, all usable. Peak memory 19.5 GB per solve (Arcturus).

**The labels the families share do as well or better.** Against the same references the CNO network's
Teff, logg and [M/H] cells pass G1 more often than the five-label network's (9, 13 and 13 of 16 cells
at the Sun, giant and Arcturus, against 7, 9 and 8, counting its own [α/M] cells). Its value errors are
2–4× smaller in T and up to 4× smaller in P_gas; in m and n_e they are mixed.

**The C, N and O derivatives of the atmosphere are not usable.** Central ±h rel-L2 (cosine) / reference
spread, over −3 ≤ log τ ≤ 1 (`results/atmosphere_jacobian_cno8.json`); bold where the error is more
than twice the spread:

| cell | Sun | giant | Arcturus |
|---|---|---|---|
| T / [C/M] | 0.28 (0.97) / 0.42 | 0.47 (0.88) / 1.09 | 0.21 (0.99) / 0.15 |
| P_gas / [C/M] | **0.38** (0.97) / 0.15 | **1.13** (0.98) / 0.35 | 0.46 (0.95) / 0.34 |
| m / [C/M] | 0.23 (0.99) / 0.12 | **0.32** (0.98) / 0.09 | 0.21 (0.98) / 0.11 |
| n_e / [C/M] | 0.54 (0.88) / 0.32 | 0.95 (0.36) / 1.71 | 0.27 (0.98) / 0.24 |
| T / [N/M] | 0.32 (0.98) / 0.70 | 0.80 (0.60) / 2.49 | 0.66 (0.76) / 1.36 |
| P_gas / [N/M] | 0.27 (0.97) / 0.45 | 0.77 (0.78) / 1.82 | 0.88 (0.51) / 0.84 |
| m / [N/M] | **0.64** (0.85) / 0.08 | **1.82** (0.96) / 0.23 | **2.00 (−0.98)** / 0.11 |
| n_e / [N/M] | **1.07** (0.90) / 0.48 | 1.62 (−0.05) / 3.17 | 1.15 (0.87) / 1.11 |
| T / [O/M] | 0.27 (0.97) / 0.60 | 0.47 (0.89) / 1.08 | 0.53 (0.88) / 1.37 |
| P_gas / [O/M] | **2.85 (−0.81)** / 0.72 | **4.23** (0.88) / 1.48 | 1.45 (0.28) / 0.86 |
| m / [O/M] | **1.02** (0.97) / 0.13 | **1.46 (−0.49)** / 0.21 | **0.64** (0.86) / 0.23 |
| n_e / [O/M] | **3.51 (−0.86)** / 1.06 | 1.84 (0.04) / 1.96 | 1.26 (0.55) / 1.38 |

Several cells have the **wrong sign** (negative cosine), and none passes G1. The cells not in bold are
mostly unresolvable rather than good: the reference spread is as large as the signal. This is DSS's
mechanism at its most extreme. A 0.05 dex change in C, N or O moves the atmosphere far less than the
emulator's own value error (about 1% in P_gas), so a network trained on values leaves these slopes
essentially unconstrained. The [α/M] cells, with O fixed, are fine at the giants (P_gas and m pass G1)
and 6–11% off at the Sun.

**Flux Jacobian, H band** (rel-L2; cosine in parentheses where it is below 0.998; atmosphere share and
the emulator's error on the atmosphere part from `flux_split.py`):

| label | Sun | giant | Arcturus |
|---|---|---|---|
| Teff | 0.025 | 0.009 | 0.015 |
| logg | 0.026 | 0.023 | 0.014 |
| [M/H] | 0.009 | 0.004 | 0.005 |
| [α/M], O fixed | 0.077 (0.9979) | 0.024 | 0.031 |
| [C/M] | **0.106** (0.9946); share 21%, atm. part 0.50 | 0.083 (0.9968); share 32%, atm. part 0.26 | 0.008; share 3%, atm. part 0.26 |
| [N/M] | 0.041; share 7%, atm. part 0.58 | **0.159** (0.9872); share 9%, atm. part 1.70 | 0.040; share 2%, atm. part 2.29 |
| [O/M] | **0.205** (0.9801); share 6%, atm. part 3.60 | 0.039; share 1%, atm. part 3.11 | 0.018; share 1%, atm. part 1.78 |

**Reading.**
- **Teff, logg and [M/H] flux Jacobians are 0.4–2.6%**, better than the five-label network's (§6).
- **C, N and O act on the H band mostly directly**, through the CO, CN and OH line opacity, which does
  not involve the emulator. The atmosphere carries 1–32% of the response, and on that part the emulator
  is 26–360% off. The flux error is therefore small where the atmosphere's share is small (Arcturus:
  0.8–4%) and large where it is not: the Sun's [O/M] (21%), the giant's [N/M] (16%) and the Sun's
  [C/M] (11%).
- **For a fit:** use the CNO network's tangent for Teff, logg, [M/H] and [α/M], but take the
  atmosphere's response to C, N and O from finite differences, or neglect it (set it to zero), rather
  than from the emulator. Neglecting it costs the atmosphere share above (1–32%); using the emulator can
  cost several times that, with the wrong sign.
- Only the H band was measured for this family.

---

## 7. What this means for DSS

Ranked by how directly the measurements support each option:

1. **Hybrid seam (recommended).** Keep DSS's `AtlasResolver` for the value, with ATLAS12 or the Payne
   Zero solver. Replace the tangent rule in `state_fn`'s `custom_jvp` with the emulator's autodiff
   Jacobian at the current labels. Recompute the finite-difference seam Jacobian once at convergence
   for uncertainties.
   - **Why it's safe:** the fixed point of Gauss–Newton or Levenberg–Marquardt depends only on exact
     residuals. A Jacobian that is 2–10% off, with cosine ≥ 0.997, changes the path and iteration
     count, not the answer.
   - **What it saves:** the nine re-solves per Jacobian, which are DSS's dominant cost per step.
   - **CNO labels:** with the CNO initializer, take the atmosphere's response to C, N and O from
     finite differences or neglect it; its tangent for them is wrong (§6.3).
   - **Integration:** `AtmosphereInitializer.log_state_jacobian` already returns d ln(T, P_gas, m, n_e)/d label
     on DSS's grid. DSS's tangent is in linear units, so multiply by the state value.

2. **Emulator for both value and gradient (fully differentiable fast path).** Useful for exploration,
   initialization and sampling-based work at moderate accuracy. It inherits the emulator's value
   errors: about 2–3% in P_gas and about 0.06% in T. DSS's §5 shows these slope errors are smooth over
   hundreds of kelvin, so they bias the labels rather than adding noise.

3. **Not recommended:** emulator derivatives as the final Jacobian for uncertainties. [α/M] (4–11%) and
   logg (up to 12%) are too far off across whole windows, strong-line cores are 10–60% off (§6.2), and
   the cool-star regime is sampled by only one star.

**Separate observation.** Payne Zero's solver is a reimplementation of ATLAS12 with the same
abundance conventions as DSS. Here it re-converged a model in 3–18 minutes on 4 threads. As the value
source behind DSS's seam it would be in-process and warm-started by the same initializer. Whether it is
consistent enough with DSS's pyKurucz ATLAS12 to keep Phase 3's 0.06 K seam budget would have to be
re-measured.

---

## 8. Limitations

1. **Three stars, six windows.** The Sun, one metal-poor giant and Arcturus, in 420–440, 510–520,
   846–870, 1170–1330 (J), 1550–1600 (H) and 2000–2300 nm (Ks). That is not a survey of the label space or of wavelength; the UV and
   strong-line regions other than Mg b and the Ca II triplet are unmeasured. All synthesis is LTE.
2. **One cool star.** DSS found its emulator's errors worst toward cool, molecule-rich atmospheres.
   Arcturus confirms this at the atmosphere level for [α/M] (§5.4) without it reaching the H band
   (§6). Cool dwarfs are unmeasured; a `kdwarf` base (4500 K, logg 4.6) is defined in
   `fd_reference.py` but was not run.
3. **The reference is mostly Payne Zero's own solver.** The emulator was trained to imitate it, which
   favours it. §5.5 checks against ATLAS12 at five bases, at the atmosphere level only, with looser
   ATLAS12 convergence; the bias is small there. Flux-level and CNO results are against Payne Zero only.
4. **The giant bases are not verified hold-outs.** The Sun matches an excluded reference star. The
   metal-poor giant is an arbitrary in-support point and may lie near training models. Arcturus is not
   one of the excluded stars either; the nearest one is 214 K hotter and 0.34 dex higher in logg.
5. **Incomplete convergence.** Of the 51 solves, 37 met the 5×10⁻⁶ target. Ten more stopped at the
   iteration cap between 5×10⁻⁶ and 1.7×10⁻⁵, still at least 30× tighter than production. Four are
   unusable: Teff + 2h at the Sun and the giant, the giant's [M/H] − 2h, and Arcturus's [α/M] − 2h,
   between 1.1×10⁻⁴ and 7.7×10⁻⁴. All four are 2h points. They degrade only the Richardson column, where the stability filter drops the
   affected layers. The central ±h references and all flux tests use only ±h solves, and each of those
   converged or stopped below 2×10⁻⁵.
6. **Weak-signal cells.** T against logg, [M/H] and [α/M] are unresolved at the atmosphere level
   because the true response is smaller than the finite-difference noise at these steps. Strong-line
   cores (§6.2) indirectly show the emulator's ∂T/∂logg and ∂T/∂[M/H] to be wrong at the Sun.
7. **Not the direct-abundance initializer.** The five-label and CNO initializers were tested, the CNO
   one in the H band only (§6.3). The direct-abundance initializer was not. Its mode, with about 80 [X/Fe] labels, is where autodiff would save the most
   (about 80 solves per Jacobian), and it is unmeasured.
8. **The hybrid seam itself was not run inside DSS.** §7 is a recommendation derived from these
   measurements, not a demonstrated DSS fit.

---

## 9. Reproducing

**Requirements.**
- **The validation** (not atmojax itself) needs a working Payne Zero, for the solver and the synthesis:
  - `pip install -e .` in a Payne Zero checkout;
  - its runtime data, `git lfs pull --include="source_data_files/**"` (about 7 GB), then verify with
    `python payne_zero_atmosphere/install_runtime_data.py --manifest source_data_files/runtime_data_manifest.json verify --root source_data_files`;
  - `export PAYNE_ZERO_DATA_ROOT=<checkout>/source_data_files`.
- `pip install -e ".[validation]"` in this repository (adds torch and scipy).
- About 15 GB of RAM for the Sun and the metal-poor giant; about 20 GB per Arcturus solve (§4.3).
- About 5 GB of free disk for `validation/work/`.

```bash
cd validation
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4

python parity.py                             # work/five_label.npz + results/parity.json
python fd_reference.py build-catalog         # work/predicted_atomic_lines_all.npy (4.2 GB)

python fd_reference.py solve --base sun              # 17 solves, ~80 min on 4 threads; resumable
python fd_reference.py solve --base metalpoor_giant  # 17 solves, ~100 min
python fd_reference.py solve --base arcturus         # 17 solves, ~3 h; or split with --only (§4.3)

python compare.py results/runs/sun results/runs/metalpoor_giant results/runs/arcturus  # results/atmosphere_jacobian.json
python flux_jacobian.py results/runs/sun --wl 1550 1600 --r-grid 100000
python flux_jacobian.py results/runs/metalpoor_giant --wl 1550 1600 --r-grid 100000
python flux_jacobian.py results/runs/arcturus --wl 1550 1600 --r-grid 100000
python flux_split.py results/runs/sun                # after flux_jacobian.py: reads its flux_jac_*.npz
python flux_split.py results/runs/metalpoor_giant
python flux_split.py results/runs/arcturus
for band in "420 440" "510 520" "846 870" "1170 1330" "2000 2300"; do for base in sun metalpoor_giant arcturus; do
  python flux_jacobian.py results/runs/$base --wl $band   # outputs suffixed _<band>nm
  python flux_split.py results/runs/$base --wl $band
done; done
python flux_cores.py results/runs/sun                # after the band runs above; one base per call
python flux_cores.py results/runs/metalpoor_giant
python flux_cores.py results/runs/arcturus

# CNO initializer (§6.3): needs work/cno8.npz from `atmojax-export cno8 work/cno8.npz`
for base in sun metalpoor_giant arcturus; do
  python fd_reference.py solve --family cno8 --base $base   # 17 new solves; reuses runs/<base> teff/logg/mh
done
python compare.py results/runs_cno8/sun results/runs_cno8/metalpoor_giant results/runs_cno8/arcturus
for base in sun metalpoor_giant arcturus; do
  python flux_jacobian.py results/runs_cno8/$base && python flux_split.py results/runs_cno8/$base
done
python checks.py                             # results/checks.json (DSS checkout optional, via $DSS_REPO)
python atlas12_compare.py                    # results/atlas12_jacobian.json (needs the DSS checkout's artifacts)
```

The recorded solves already exist under `results/runs/`. The analysis steps (`compare.py` onward) can
be re-run without re-solving. `flux_*` still need the runtime data for synthesis.

**Provenance.** This study was first run as `experiments/differentiable_initializer/` on the
`ccr-3759fd72-bop93g` branch of `leejjoon/payne-zero`, with a standalone `pz_jax_init.py`. When it
moved here, `atmojax.AtmosphereInitializer` was checked against that original port. It reproduces the
state and the label Jacobian with zero difference at the Sun and the giant, and
`results/atmosphere_jacobian.json` regenerates with zero difference. Every number in this report
therefore applies unchanged to the package.

---

## 10. Files (under `validation/`)

| path | contents |
|---|---|
| `common.py` | shared paths; loads the atmojax initializer under test |
| `parity.py` | exports the weights to `work/`; port parity at five labels |
| `fd_reference.py` | reference solves; `build-catalog`; low-memory catalog patch |
| `compare.py` | atmosphere-level Jacobian scores (Richardson and central ±h with reference spread) |
| `flux_jacobian.py` | spectrum-level Jacobian test (reference, emulator, T-only arms) |
| `flux_split.py` | direct versus atmosphere split for [M/H] and [α/M] |
| `flux_cores.py` | flux-Jacobian scores on strong-line core pixels, with the reference's one-sided spread |
| `checks.py` | τ-grid alignment, tolerance sensitivity, DSS convention identity |
| `atlas12_compare.py` | §5.5: scores against DSS's stored ATLAS12 V1.3 solves, and head to head with DSS's emulator |
| `results/atlas12_jacobian.json` | §5.5 |
| `results/parity.json` | port parity at five labels |
| `results/atmosphere_jacobian.json` | §5 tables, including depth-resolved errors |
| `results/checks.json` | §3 and §4 supporting checks |
| `results/runs/<base>/<label>_{p,m}{1,2}.npz`, `base.npz` | each solve: labels, convergence, diagnostics, canonical-grid and native T, P_gas, m, n_e |
| `results/runs/sun/base_tol2e-5.npz` | the Sun base at the looser tolerance (§4.2 check) |
| `results/runs/<base>/flux_jac_<label>.npz` | wavelength and the three flux-Jacobian arms |
| `results/runs/<base>/flux_jacobian.json`, `flux_split.json` | §6 tables (H band) |
| `results/runs/<base>/flux_jac_<label>_<band>nm.npz`, `flux_jacobian_<band>nm.json`, `flux_split_<band>nm.json` | §6.1 (other bands) |
| `results/runs/<base>/flux_cores.json` | §6.2 |
| `results/runs_cno8/<base>/` | §6.3: CNO-family solves (base, am, cm, nm, om), flux Jacobians and split |
| `results/atmosphere_jacobian_cno8.json` | §6.3 atmosphere table |
| `results/logs/` | solver and synthesis logs; `mem.log` holds the memory trace of the patched run. `campaign.log` also contains the out-of-memory Arcturus attempt; `arcturus_*.log` are the later Arcturus solves (with `/usr/bin/time -v` peak memory), `arcturus_mem.log` their machine-wide memory trace, and `flux_arcturus.log`/`flux_split_arcturus.log` its synthesis |
| `work/` (git-ignored) | exported weights and the combined catalog, regenerated by the commands above |
