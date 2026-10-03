"""Score atmojax's label Jacobian against ATLAS12 instead of Payne Zero's own solver.

DSS's Phase 1 V1.3 campaign (scripts/phase1_v13_jacobians.py in differentiable_stellar_spectroscopy)
stored ATLAS12 (pyKurucz) solves at +-h, +-2h, +-4h in Teff, logg, [M/H] and [alpha/M] around 12 base
labels, with the same steps as fd_reference.py.  Its decks are read here, put on the canonical tau grid
exactly as fd_reference.py does, and scored with compare.py's metrics.  The initializer was trained on
Payne Zero's solver, so this removes the reference's bias in its favour.  Bases outside the
initializer's training box are skipped.  No solves are run.

The same solves also scored DSS's own Kurucz-a1 emulator (fd/v1_3_report.json, the measurement behind
its Gate G1).  atmojax is rescored here with that report's band and stability threshold, cell by cell,
for a like-for-like comparison ("dss_v13_head_to_head" in the output).

    python atlas12_compare.py                 # DSS checkout at $DSS_REPO, else ../../differentiable_stellar_spectroscopy
"""
from __future__ import annotations

import json, os, re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import compare as C
import fd_reference as F
from common import HERE, RESULTS, initializer

DSS = Path(os.environ.get("DSS_REPO", HERE.parent.parent / "differentiable_stellar_spectroscopy"))
FD = DSS / "artifacts" / "phase1" / "fd"
DSS_FIELD = {"T": "T", "P_gas": "P", "m": "RHOX", "n_e": "XNE"}
GRID = DSS / "artifacts" / "phase1" / "grid" / "models"


def read_deck(path: Path):
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("READ DECK6"))
    n = int(lines[i].split()[2])
    a = np.array([[float(x) for x in l.split()[:7]] for l in lines[i + 1:i + 1 + n]])
    atm = SimpleNamespace(column_mass=a[:, 0], temperature=a[:, 1], gas_pressure=a[:, 2],
                          electron_density=a[:, 3], rosseland_opacity=a[:, 4])
    return F.to_tau_grid(atm), float(np.median(a[:, 6])) / 1e5


def last_dlnt(model_dir: Path):
    r = json.loads((model_dir / "result.json").read_text())
    m = re.search(r"checkconv_dlnt=([0-9.eE+-]+)", r.get("iterations_log", {}).get("last_iteration_line", ""))
    return bool(r.get("ok")), float(m.group(1)) if m else float("nan")


def log_state(g):
    return np.log(np.stack([g[f] for f in C.FIELDS]))


def g1(c):
    return "rel_l2" in c and c["rel_l2"] <= 0.05 and c["cosine"] >= 0.995


def main():
    plan = json.loads((FD / "fd_plan.json").read_text())
    v13 = json.loads((FD / "v1_3_report.json").read_text())
    dss = {b["key"]: b["labels_ok"] for b in v13["bases"]}
    h2h = {"band_log_tau": v13["band"], "stability_threshold": v13["stability_tol"], "bases": {}}
    assert plan["h0"] == {k: F.STEPS[k] for k in F.LABELS}, "DSS steps differ from fd_reference.STEPS"
    ji = initializer()
    full = {"source": str(FD), "band_log_tau": C.BAND, "stability_threshold": C.STABILITY, "bases": {}}
    for b in plan["bases"]:
        key = b["key"]
        g0, xi = read_deck(GRID / key / "model.atm")
        labels = np.array([b["teff"], b["logg"], b["mh"], b["am"], xi])
        if not ji.in_support(labels):
            print(f"skip {key}: outside the initializer's training box")
            continue
        s0 = log_state(g0)
        conv = {}

        def states(lab):
            out = []
            for n in ("p1", "m1", "p2", "m2"):
                d = FD / "models" / f"{key}__{lab}_{n}"
                conv[f"{lab}_{n}"] = last_dlnt(d)
                out.append(log_state(read_deck(d / "model.atm")[0]))
            return s0, out

        J_emu = np.asarray(ji.log_state_jacobian(labels))
        rep = C.score(J_emu, F.LABELS, states)
        band = (F.LOG_TAU >= C.BAND[0]) & (F.LOG_TAU <= C.BAND[1])
        v = np.asarray(ji.state(labels))
        vals = {fld: float(np.median(np.abs(np.log(v[f, band]) - s0[f, band]))) for f, fld in enumerate(C.FIELDS)}
        ours = C.score(J_emu, F.LABELS, states, v13["band"], v13["stability_tol"])
        cells = {}
        for lab in F.LABELS:
            for fld in C.FIELDS:
                a, d = ours[f"{fld}/{lab}"]["richardson"], dss[key][lab].get(DSS_FIELD[fld], {})
                cells[f"{fld}/{lab}"] = {"atmojax": a.get("rel_l2"), "dss_emulator": d.get("rel_l2"),
                                         "atmojax_g1": g1(a), "dss_emulator_g1": g1(d)}
        h2h["bases"][key] = cells
        full["bases"][key] = {"labels": labels.tolist(), "split": b["split"], "value_median_abs_dln": vals,
                              "solves_ok_and_final_dlnt": conv, "jacobian": rep}
        dl = [v for ok, v in conv.values()]
        print(f"\n== {key} ({b['split']})  xi {xi:g}  ATLAS12 final dlnT {np.nanmin(dl):.0e}..{np.nanmax(dl):.0e}"
              f"  all ok: {all(ok for ok, _ in conv.values())}")
        print("   emulator value error vs ATLAS12, median |d ln|: " + "  ".join(f"{k} {x:.1e}" for k, x in vals.items()))
        print(f"{'cell':12s} | {'Richardson':^26s} | {'central +-h':^30s}")
        for k, c in rep.items():
            r, h = c["richardson"], c["central_h"]
            rr = (f"{r['rel_l2']:6.3f} {r['cosine']:7.4f} {r['n_trusted_layers']:3d} {'PASS' if r['passes_G1'] else 'fail':>5s}"
                  if "rel_l2" in r else f"{'(' + str(r['n_trusted_layers']) + ' layers)':>26s}")
            print(f"{k:12s} | {rr} | {h['rel_l2']:6.3f} {h['cosine']:7.4f} {h['one_sided_spread']:11.3f}")
    both = [c for b in h2h["bases"].values() for c in b.values() if c["atmojax"] is not None and c["dss_emulator"] is not None]
    a, d = np.array([c["atmojax"] for c in both]), np.array([c["dss_emulator"] for c in both])
    h2h["summary"] = {"cells_scored_by_both": len(both), "atmojax_better": int((a < d).sum()),
                      "median_rel_l2": {"atmojax": float(np.median(a)), "dss_emulator": float(np.median(d))},
                      "max_rel_l2": {"atmojax": float(a.max()), "dss_emulator": float(d.max())},
                      "g1_passes": {k: sum(c[f"{k}_g1"] for b in h2h["bases"].values() for c in b.values())
                                    for k in ("atmojax", "dss_emulator")},
                      "cells_per_base": 4 * len(F.LABELS)}
    full["dss_v13_head_to_head"] = h2h
    print("\nvs DSS's Kurucz-a1 emulator on its V1.3 criteria: " + json.dumps(h2h["summary"]))
    out = RESULTS / "atlas12_jacobian.json"
    out.write_text(json.dumps(full, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
