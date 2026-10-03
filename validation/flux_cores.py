"""Flux-Jacobian error restricted to strong-line cores, which form higher than the band as a whole.

A window-wide rel-L2 is dominated by weak lines and wings.  Here the base spectrum is synthesized once,
each strong line's minimum is located within +-0.5 nm of its nominal wavelength (so air/vacuum does
not matter), its core is the contiguous pixels whose depth is >= CORE_DEPTH of the minimum's, and the
saved flux_jac_<label><band>.npz arms from flux_jacobian.py are scored on the union of those pixels.
A core holds few pixels and a saturated core responds weakly, so the reference's own noise is reported
too, as in compare.py: the disagreement of the one-sided differences (f(+h) - f0)/h and (f0 - f(-h))/h,
relative to the central difference.  An emulator error comparable to it is not a measured failure.

    python flux_cores.py results/runs/sun        # writes results/runs/sun/flux_cores.json
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

import fd_reference as F
import flux_jacobian as FJ

CORE_DEPTH = 0.8
LINES = {   # band -> nominal strong-line wavelengths (nm, air)
    (846.0, 870.0): {"Ca II triplet": [849.80, 854.21, 866.21]},
    (510.0, 520.0): {"Mg I b": [516.73, 517.27, 518.36]},
    (1170.0, 1330.0): {"K I (J)": [1243.22, 1252.21]},
    (2000.0, 2300.0): {"Na I (Ks)": [2205.6, 2208.4]},
}


class _Args:
    def __init__(self, wl, r_grid=100000.0):
        self.wl, self.r_grid = wl, r_grid


def core_mask(wl, f, centres):
    m = np.zeros(wl.size, bool)
    for c in centres:
        w = np.flatnonzero(np.abs(wl - c) < 0.5)
        i = w[np.argmin(f[w])]
        lo = hi = i
        thr = CORE_DEPTH * (1.0 - f[i])
        while lo > 0 and 1.0 - f[lo - 1] >= thr:
            lo -= 1
        while hi < wl.size - 1 and 1.0 - f[hi + 1] >= thr:
            hi += 1
        m[lo:hi + 1] = True
    return m


def main(dirs):
    out = {}
    for d in map(Path, dirs):
        labels = F.BASES[d.name]
        base = FJ.native(np.load(d / "base.npz"))
        ab = FJ.abundances(np.load(d / "base.npz")["labels"])
        for band, groups in LINES.items():
            a = _Args(band)
            wl, f = FJ.flux(base, ab, labels[4], a)
            sfx = FJ.band_suffix(band)
            for name, centres in groups.items():
                m = core_mask(wl, f, centres)
                rec = {"n_pix": int(m.sum()), "min_flux": [float(f[np.abs(wl - c) < 0.5].min()) for c in centres]}
                for lab in F.LABELS:
                    z = np.load(d / f"flux_jac_{lab}{sfx}.npz")
                    assert np.allclose(z["wl"], wl)
                    h = F.STEPS[lab]
                    zp, zm = np.load(d / f"{lab}_p1.npz"), np.load(d / f"{lab}_m1.npz")
                    _, fp = FJ.flux(FJ.native(zp), FJ.abundances(zp["labels"]), labels[4], a)
                    _, fm = FJ.flux(FJ.native(zm), FJ.abundances(zm["labels"]), labels[4], a)
                    repro = float(np.linalg.norm((fp - fm) / (2 * h) - z["ref"]) / np.linalg.norm(z["ref"]))
                    assert repro < 1e-3, f"{lab}: reference arm not reproduced (rel {repro:.1e})"
                    r, e, t = z["ref"][m], z["emu"][m], z["t_only"][m]
                    spread = np.linalg.norm(((fp - f) / h)[m] - ((f - fm) / h)[m]) / np.linalg.norm(r)
                    rec[lab] = {"one_sided_spread": float(spread), "reference_reproduction_rel": repro,
                                "rms_core_over_rms_band": float(np.sqrt(np.mean(r ** 2) / np.mean(z["ref"] ** 2))),
                                "rel_l2_emulator": float(np.linalg.norm(e - r) / np.linalg.norm(r)),
                                "cosine_emulator": float(e @ r / (np.linalg.norm(e) * np.linalg.norm(r))),
                                "projected_gain": float(e @ r / (r @ r)),
                                "rel_l2_T_only_emulator": float(np.linalg.norm(t - r) / np.linalg.norm(r))}
                out.setdefault(d.name, {})[name] = rec
                print(f"{d.name}/{name} ({rec['n_pix']} px): " + "  ".join(
                    f"{lab} {rec[lab]['rel_l2_emulator']:.3f}/{rec[lab]['cosine_emulator']:.4f} "
                    f"(T-only {rec[lab]['rel_l2_T_only_emulator']:.3f}, spread {rec[lab]['one_sided_spread']:.3f}, repro {rec[lab]['reference_reproduction_rel']:.0e})"
                    for lab in F.LABELS), flush=True)
        (d / "flux_cores.json").write_text(json.dumps({"core_depth": CORE_DEPTH, **out[d.name]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
