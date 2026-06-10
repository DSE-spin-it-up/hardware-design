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
from sizing.elevator import ElevatorInputs
from sizing.fuselage import FuselageInputs
from sizing.rudder import RudderInputs
from sizing.wing import SizingInputs
from structures.rods import RodInputs
from weights.mass import RibInputs
from weights.part_materials import PartMaterials


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
PROJECT_ROOT = CONFIG_PATH.parent


def _load() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _resolve_repo_path(value):
    """Anchor a relative .dat/.csv path against the repo root.

    Lets Windows + Linux teammates run `python main.py` from any cwd without
    breaking `airfoils/...` / `data/...` lookups. NACA digit strings, nulls,
    and already-absolute paths pass through untouched.
    """
    if not isinstance(value, str):
        return value
    suffix = value.lower()
    if not (suffix.endswith(".dat") or suffix.endswith(".csv")):
        return value
    p = Path(value)
    if p.is_absolute():
        return value
    return str((PROJECT_ROOT / p).resolve())


_data = _load()

if "sizing" in _data:
    for _boom_alias in ("lboom", "l_boom"):
        if _boom_alias in _data["sizing"]:
            _data["sizing"]["L_boom"] = _data["sizing"].pop(_boom_alias)
    if "rudder" in _data and "Vgust" in _data["rudder"]:
        _data["sizing"].setdefault("gust_speed", _data["rudder"].pop("Vgust"))

if "fuselage" in _data:
    _fuselage_aliases = {
        "pvc_snap_force": "tube_snap_force",
        "pvc_snap_safety_factor": "tube_snap_safety_factor",
        "pvc_gust_safety_factor": "tube_structural_safety_factor",
        "tube_gust_safety_factor": "tube_structural_safety_factor",
        "min_pvc_foam_floor": "min_tube_foam_floor",
    }
    for _old, _new in _fuselage_aliases.items():
        if _old in _data["fuselage"]:
            _data["fuselage"].setdefault(_new, _data["fuselage"].pop(_old))

_data["airfoil"] = _resolve_repo_path(_data.get("airfoil"))
_data["tail_airfoil"] = _resolve_repo_path(_data.get("tail_airfoil"))
_data["propulsion"]["csv_prop"] = _resolve_repo_path(_data["propulsion"]["csv_prop"])

MATERIALS = PartMaterials.from_names(_data["materials"])

SIZING = SizingInputs(**_data["sizing"])
PROPULSION = PropulsionInputs(**_data["propulsion"])
FUSELAGE = FuselageInputs(**_data["fuselage"])
CONTROL_SURFACE = AileronInputs(**_data["control_surface"])
ELEVATOR = ElevatorInputs(**_data.get("elevator", {}))
RUDDER = RudderInputs(**_data.get("rudder", {}))
STRUCTURE = RodInputs(**_data["structure"], material=MATERIALS.rod)
RIBS = RibInputs(**_data.get("ribs", {}))

AIRFOIL: str | None = _data["airfoil"]
TAIL_AIRFOIL: str = _data["tail_airfoil"]

BATTERY_X: float | None = _data.get("battery_x")
BATTERY_Y0: float | None = _data.get("battery_y0")
BATTERY_Y_FRAC: float | None = _data.get("battery_y_frac")
BATTERY_Y_MODE: str = _data.get("battery_y_mode", "low")
BATTERY_Y_SAMPLES: int = int(_data.get("battery_y_samples", 9))
PAYLOAD_MAX_TENSION: float | None = _data.get("payload_max_tension")

V_STALL: float = _data["V_stall"]

ALPHA_SWEEP_DEG = tuple(_data["alpha_sweep_deg"])
ALPHA_SWEEP_LOOP_DEG = tuple(_data["alpha_sweep_loop_deg"])

MASS_CLOSURE: bool = _data["mass_closure"]
SW_CLOSURE: bool = _data["sw_closure"]
N_ITER_MAX: int = _data["n_iter_max"]
CD0_TOL: float = _data["cd0_tol"]
MASS_TOL: float = _data["mass_tol"]
SW_TOL: float = _data["sw_tol"]
