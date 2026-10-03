"""Finite-difference label Jacobians of Payne Zero's converged atmosphere solver.

For each base: solve at the base and at +-h, +-2h in each label (fresh emulator start, tight
convergence), keep the float64 in-memory state, put T, P_gas, m, n_e on the standard tau_Ross grid
(PCHIP in log tau of the model's own integrated Rosseland depth, as DSS does), save everything.

    python fd_reference.py build-catalog              # once: work/predicted_atomic_lines_all.npy (4.2 GB)
    python fd_reference.py solve --base sun           # resumable; results/runs/sun/<name>.npz per solve
    python fd_reference.py solve --base sun --family cno8 --only base,cm_p1,...   # results/runs_cno8/sun/

The cno8 family holds [C/M], [N/M] and [O/M] as labels.  Its base has [C/M] = [N/M] = 0 and
[O/M] = [alpha/M], which is exactly the five-label abundance mixture (Payne Zero's five-label path
treats O as an alpha element), so the five-label teff/logg/mh solves are reused as its references
(see run_dir).  Its alpha perturbation holds [O/M] fixed, unlike the five-label one, so it is re-solved.

The recorded campaign used --tol 5e-6 --max-iter 60 (the defaults below), NUMBA_NUM_THREADS=4.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

from common import RESULTS, WORK, payne_zero_data_root

COMBINED_PREDICTED = WORK / "predicted_atomic_lines_all.npy"

BASES = {
    "sun": (5777.0, 4.44, 0.0, 0.0, 1.0),
    "arcturus": (4286.0, 1.66, -0.52, 0.30, 1.7),
    "kdwarf": (4500.0, 4.60, 0.0, 0.0, 1.0),
    "metalpoor_giant": (5000.0, 2.50, -1.50, 0.40, 1.5),
}
LABELS = ("teff", "logg", "mh", "am")
STEPS = {"teff": 25.0, "logg": 0.05, "mh": 0.05, "am": 0.05, "cm": 0.05, "nm": 0.05, "om": 0.05}
# position in the stored label vector (teff, logg, mh, am, vmic[, cm, nm, om]), which is also the column
# of atmojax's log_state_jacobian for both families
LABEL_INDEX = {"teff": 0, "logg": 1, "mh": 2, "am": 3, "cm": 5, "nm": 6, "om": 7}
FAMILIES = {"five_label": LABELS, "cno8": LABELS + ("cm", "nm", "om")}
RUNS = {"five_label": RESULTS / "runs", "cno8": RESULTS / "runs_cno8"}
REUSED_FROM_FIVE_LABEL = ("teff", "logg", "mh")


def base_labels(base: str, family: str = "five_label") -> tuple:
    b = BASES[base]
    return b if family == "five_label" else b + (0.0, 0.0, b[3])


def family_of(d: Path) -> str:
    return "cno8" if d.parent.name == RUNS["cno8"].name else "five_label"


def run_dir(d: Path, lab: str) -> Path:
    """Directory holding the solves for label `lab` at base directory `d` (cno8 reuses five-label ones)."""
    if family_of(d) == "cno8" and lab in REUSED_FROM_FIVE_LABEL and not (d / f"{lab}_p1.npz").exists():
        return RUNS["five_label"] / d.name
    return d
LOG_TAU = -6.875 + 0.125 * np.arange(80)


def perturbations(family: str = "five_label"):
    yield "base", None, 0
    for lab in FAMILIES[family]:
        i = LABEL_INDEX[lab]
        for mult in (1, 2):
            for sgn in (+1, -1):
                yield f"{lab}_{'p' if sgn > 0 else 'm'}{mult}", i, sgn * mult


def to_tau_grid(atm):
    from scipy.interpolate import PchipInterpolator
    m = np.asarray(atm.column_mass, float)
    k = np.asarray(atm.rosseland_opacity, float)
    # trapezoid tau_Ross with the outermost layer's tau = m0 * k0 (ATLAS convention)
    tau = np.concatenate([[m[0] * k[0]], m[0] * k[0] + np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(m))])
    lt = np.log10(tau)
    out = {"native_log_tau": lt}
    for name, arr in (("T", atm.temperature), ("P_gas", atm.gas_pressure), ("m", m), ("n_e", atm.electron_density)):
        out[name] = 10 ** PchipInterpolator(lt, np.log10(np.asarray(arr, float)), extrapolate=True)(LOG_TAU)
        out[name + "_native"] = np.asarray(arr, float)
    return out


def build_combined_catalog() -> None:
    """Concatenate the three predicted-line shards, in order, into one memory-mappable .npy."""
    parts = sorted((payne_zero_data_root() / "source_catalogs" / "lines").glob("predicted_atomic_lines_part*.npy"))
    maps = [np.load(p, mmap_mode="r") for p in parts]
    WORK.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(COMBINED_PREDICTED, mode="w+", dtype=maps[0].dtype,
                                    shape=(sum(m.shape[0] for m in maps),) + maps[0].shape[1:])
    i = 0
    for m in maps:
        for s in range(0, m.shape[0], 10_000_000):
            e = min(s + 10_000_000, m.shape[0])
            out[i + s:i + e] = m[s:e]
        i += m.shape[0]
    out.flush()
    print(f"wrote {COMBINED_PREDICTED} from {[p.name for p in parts]}")


def _patch_low_memory_catalog_reads():
    """Run the solver in a 15 GB container.  Stock Payne Zero keeps every source catalog resident
    in RAM (and np.concatenate over the 3 x 1.4 GB predicted-line shards alone peaks near 8.4 GB),
    which the out-of-memory killer stopped twice.  Here the standard and diatomic catalogs are
    memory-mapped instead: identical bytes, file-backed pages.  Experiment-local; no repo change."""
    import payne_zero_atmosphere.line_selection as LS
    combined = COMBINED_PREDICTED
    if not combined.exists():
        raise FileNotFoundError(f"{combined} missing: run `python fd_reference.py build-catalog` first")

    def mapped(path):
        arr = np.load(path, mmap_mode="r").view(np.ndarray)
        return arr.view(np.int32).reshape(-1, 4) if arr.dtype.fields is not None else arr

    def standard(path):
        p = Path(path)
        if p.suffix != ".npy":
            return original_standard(path)
        return mapped(combined if p.stem.endswith("_part1") else p)

    def diatomic(path):
        return mapped(path) if Path(path).suffix == ".npy" else original_diatomic(path)

    original_standard = LS._read_standard_line_catalog_uncached
    original_diatomic = LS._read_diatomic_line_catalog_uncached
    LS._read_standard_line_catalog_uncached = standard
    LS._read_diatomic_line_catalog_uncached = diatomic


def solve_one(labels, tol, max_iter):
    _patch_low_memory_catalog_reads()
    from payne_zero_atmosphere.config import AtmosphereConfig, AtmosphereInput, AtmosphereOutput
    from payne_zero_atmosphere.runner import run_atmosphere_model
    from payne_zero_atmosphere.warm_start import emulator_warm_start_model
    from payne_zero_atmosphere.cli import molecular_equilibrium_catalog_path, source_line_paths
    teff, logg, mh, am, xi = labels[:5]
    cno = dict(zip(("carbon_enhancement", "nitrogen_enhancement", "oxygen_enhancement"), labels[5:]))
    warm, _ = emulator_warm_start_model(effective_temperature=teff, log_surface_gravity=logg, metallicity=mh,
                                        alpha_enhancement=am, microturbulence_km_s=xi, device="cpu", **cno)
    cfg = AtmosphereConfig(
        inputs=AtmosphereInput(initial_atmosphere=warm, molecules_path=molecular_equilibrium_catalog_path(), **source_line_paths()),
        outputs=AtmosphereOutput(),
        iterations=max_iter, enable_molecules=True, enable_convection=True, enable_convergence_stop=True,
        minimum_iterations_before_convergence=5, required_consecutive_converged_iterations=2,
        maximum_deep_layer_relative_temperature_change=tol, maximum_all_layer_relative_temperature_change=10 * tol)
    t0 = time.time()
    res = run_atmosphere_model(cfg)
    return res, time.time() - t0


def cmd_solve(a):
    base = base_labels(a.base, a.family)
    out = Path(a.out or RUNS[a.family]) / a.base
    out.mkdir(parents=True, exist_ok=True)
    for name, idx, mult in perturbations(a.family):
        if a.only and name not in a.only.split(","):
            continue
        path = out / f"{name}.npz"
        if path.exists():
            continue
        labels = list(base)
        if idx is not None:
            labels[idx] = round(labels[idx] + mult * STEPS[name.split("_")[0]], 6)
        print(f"[{a.base}] {name} labels={labels}", flush=True)
        res, wall = solve_one(labels, a.tol, a.max_iter)
        g = to_tau_grid(res.atmosphere)
        diag = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in res.diagnostics.items()
                if isinstance(v, (int, float, str, np.floating))}
        np.savez(path, labels=np.array(labels), converged=res.converged, iterations=res.iterations_completed,
                 wall_s=wall, diagnostics=json.dumps(diag), **g)
        print(f"   converged={res.converged} iters={res.iterations_completed} wall={wall:.0f}s "
              f"dT_deep={diag.get('deep_layer_relative_temperature_change')}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-catalog")
    s = sub.add_parser("solve")
    s.add_argument("--base", required=True, choices=list(BASES))
    s.add_argument("--family", default="five_label", choices=list(FAMILIES))
    s.add_argument("--out", default="", help="default: results/runs (five_label) or results/runs_cno8")
    s.add_argument("--tol", type=float, default=5e-6)
    s.add_argument("--max-iter", type=int, default=60)
    s.add_argument("--only", default="")
    a = p.parse_args()
    if a.cmd == "build-catalog":
        build_combined_catalog()
    else:
        cmd_solve(a)
