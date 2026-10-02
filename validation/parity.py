"""Export the five-label weights to work/ and record parity with Payne Zero's own predict().

    python parity.py          # -> work/five_label.npz, results/parity.json
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from atmojax import FIELDS
from atmojax.export import default_checkpoint, export
from common import RESULTS, WEIGHTS, initializer

LABELS = [(5777, 4.44, 0.0, 0.0, 1.0), (4286, 1.66, -0.52, 0.30, 1.7),
          (5000, 2.50, -1.50, 0.40, 1.5), (9000, 4.2, 0.3, 0.0, 2.0), (4100, 1.0, -2.0, 0.4, 2.5)]


def main():
    from payne_zero_atmosphere.warm_start import load_atmosphere_initializer
    ckpt = default_checkpoint("five_label")
    export(ckpt, WEIGHTS)
    ref = load_atmosphere_initializer(checkpoint_path=ckpt, device="cpu")
    predict = jax.jit(initializer().predict)
    out = {}
    for lab in LABELS:
        r = ref.predict(effective_temperature=lab[0], log_surface_gravity=lab[1], metallicity=lab[2],
                        alpha_enhancement=lab[3], microturbulence_km_s=lab[4])
        y = np.asarray(predict(jnp.asarray(lab, float)))
        errs = {}
        for i, k in enumerate(FIELDS):
            if k == "radiative_acceleration":    # signed field: scale by its maximum magnitude
                errs[k] = float(np.max(np.abs(y[:, i] - r[k])) / np.max(np.abs(r[k])))
            else:
                errs[k] = float(np.max(np.abs(y[:, i] / r[k] - 1.0)))
        out[str(lab)] = errs
        print(lab, "  ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "parity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
