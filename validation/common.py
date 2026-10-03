"""Shared paths and the initializer under test, for the validation scripts.

The validation needs a working payne-zero install with its runtime data (for the solver and the
synthesis); atmojax itself does not.
"""
from __future__ import annotations

import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = Path(os.environ.get("ATMOJAX_VALIDATION_WORK", HERE / "work"))


def payne_zero_data_root() -> Path:
    from payne_zero_atmosphere.data_files import data_root
    return data_root()


def initializer(family: str = "five_label"):
    """The atmojax initializer under test, exactly as validated (float32 MLP, as in Payne Zero)."""
    from atmojax import AtmosphereInitializer
    weights = WORK / f"{family}.npz"
    if not weights.exists():
        how = "python parity.py" if family == "five_label" else f"atmojax-export {family} {weights}"
        raise FileNotFoundError(f"{weights} missing: run `{how}` (exports the weights) first")
    return AtmosphereInitializer(weights)
