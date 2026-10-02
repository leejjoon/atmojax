"""Export a Payne Zero initializer checkpoint (Torch ``.pt``) to an atmojax ``.npz``.

Needs ``torch``, and either a checkpoint path or an installed ``payne-zero`` whose runtime data are
present (its default checkpoint locations are used).  Run once; afterwards only JAX and NumPy are needed.

    atmojax-export five_label  five_label.npz
    atmojax-export cno8        cno8.npz
    atmojax-export --checkpoint path/to/checkpoint.pt  out.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SUPPORTED_FORMATS = ("payne_zero_complete_atmosphere_latent_v2", "payne_zero_cno8_complete_atmosphere_latent_v3")
FAMILIES = ("five_label", "cno8")


def default_checkpoint(family: str) -> Path:
    """Payne Zero's own default checkpoint location for a family (requires payne-zero installed)."""
    from payne_zero_atmosphere import warm_start as W
    return {"five_label": W.DEFAULT_FIVE_LABEL_WEIGHTS_PATH, "cno8": W.DEFAULT_CNO8_WEIGHTS_PATH}[family]


def export(checkpoint: str | Path, out: str | Path) -> Path:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported checkpoint format {ck.get('format')!r}; expected one of {SUPPORTED_FORMATS}")
    fields = list(ck["labels"]["fields"])
    if list(ck["coordinates"]["fields"]) != ["log10_column_mass_increment", "log10_temperature_relative_to_grey",
                                             "log10_gas_pressure", "log10_electron_density",
                                             "log10_rosseland_opacity", "asinh_radiative_acceleration"]:
        raise ValueError("unexpected coordinate fields; the decode in atmojax would not match")
    family = "cno8" if "cno8" in ck["format"] else "five_label"
    arrays = {
        "family": np.array(family),
        "source_format": np.array(ck["format"]),
        "feature_fields": np.array(fields),
        "label_mean": np.asarray(ck["labels"]["mean"], np.float64),
        "label_std": np.asarray(ck["labels"]["std"], np.float64),
        "label_bounds": np.asarray([ck["labels"]["bounds"][f] for f in fields], np.float64),
        "tau": np.asarray(ck["coordinates"]["standard_rosseland_optical_depth"], np.float64),
        "acceleration_scale": np.float64(ck["coordinates"]["acceleration_scale"]),
    }
    for k in ("coordinate_mean", "coordinate_std", "basis", "coefficient_mean", "coefficient_std"):
        arrays["pca_" + k] = np.asarray(ck["pca"][k], np.float64)
    sd = ck["model"]["state_dict"]
    ids = sorted({int(k.split(".")[0]) for k in sd})
    for i, lid in enumerate(ids):
        arrays[f"W{i}"] = sd[f"{lid}.weight"].double().numpy()
        arrays[f"b{i}"] = sd[f"{lid}.bias"].double().numpy()
    arrays["n_layers"] = np.int64(len(ids))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family_or_out", help="five_label | cno8 (then OUT), or OUT when --checkpoint is given")
    p.add_argument("out", nargs="?")
    p.add_argument("--checkpoint", help="explicit path to a Payne Zero checkpoint.pt")
    a = p.parse_args(argv)
    if a.checkpoint:
        ckpt, out = Path(a.checkpoint), Path(a.family_or_out)
    else:
        if a.family_or_out not in FAMILIES or not a.out:
            p.error("give FAMILY OUT (family: five_label or cno8), or --checkpoint PATH OUT")
        ckpt, out = default_checkpoint(a.family_or_out), Path(a.out)
    print(f"wrote {export(ckpt, out)} from {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
