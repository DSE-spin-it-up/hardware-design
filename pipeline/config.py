"""Load design knobs from `config.yaml` and expose them as module attributes.

The pipeline accesses these constants via the module (`config.SIZING`,
`config.PROPULSION`, ...). To tune the design, edit `config.yaml` at the
repo root — do not edit values in this file.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from propulsion.sizing import PropulsionInputs
from sizing.aileron import AileronInputs
from sizing.fuselage import FuselageInputs
from sizing.wing import SizingInputs
from structures.rods import RodInputs
from weights.part_materials import PartMaterials


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


_data = _load()

MATERIALS = PartMaterials.from_names(_data["materials"])

SIZING = SizingInputs(**_data["sizing"])
PROPULSION = PropulsionInputs(**_data["propulsion"])
FUSELAGE = FuselageInputs(**_data["fuselage"])
CONTROL_SURFACE = AileronInputs(**_data["control_surface"])
STRUCTURE = RodInputs(**_data["structure"], material=MATERIALS.rod)

AIRFOIL: str | None = _data["airfoil"]
TAIL_AIRFOIL: str = _data["tail_airfoil"]

BATTERY_X: float | None = _data.get("battery_x")

V_STALL: float = _data["V_stall"]

ALPHA_SWEEP_DEG = tuple(_data["alpha_sweep_deg"])
ALPHA_SWEEP_LOOP_DEG = tuple(_data["alpha_sweep_loop_deg"])

MASS_CLOSURE: bool = _data["mass_closure"]
SW_CLOSURE: bool = _data["sw_closure"]
N_ITER_MAX: int = _data["n_iter_max"]
CD0_TOL: float = _data["cd0_tol"]
MASS_TOL: float = _data["mass_tol"]
SW_TOL: float = _data["sw_tol"]
