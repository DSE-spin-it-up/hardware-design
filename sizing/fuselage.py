from dataclasses import dataclass, fields
import numpy as np
from sizing.initial_sizing import SizingResult


@dataclass
class FuselageInputs:
    # Battery cell envelope [m]
    battery_length: float = 0.212
    battery_width: float = 0.090
    battery_height: float = 0.060
    # Geometric margins
    housing_factor: float = 1.5  # axial battery housing factor
    casing_factor: float = 1.1   # all-around casing margin
    foam_density: float = 48     # [kg/m^3]
    casing_thickness: float = 0.003  # [m] wall thickness of casing shell


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    length: float       # [m]
    width: float        # [m]
    height: float       # [m]
    d_eq: float         # equivalent diameter [m]
    fineness: float     # length / d_eq [-]
    Swet: float         # wetted area [m^2]
    mass: float         # structural mass [kg]


def _casing_mass(length: float, width: float, height: float, inputs: FuselageInputs) -> float:
    """
    Estimate fuselage casing mass as a hollow rectangular shell.

    The shell volume is approximated as the difference between the outer box
    and the inner box (outer dimensions minus one wall thickness on each side).

    Parameters
    ----------
    length, width, height : float
        Outer fuselage dimensions [m].
    inputs : FuselageInputs
        Contains ``casing_thickness`` [m] and ``foam_density`` [kg/m³].

    Returns
    -------
    float
        Casing mass [kg].
    """
    t = inputs.casing_thickness
    # Inner dimensions (clamp to zero to avoid negative volumes on tiny fuselages)
    l_in = max(length - 2 * t, 0.0)
    w_in = max(width  - 2 * t, 0.0)
    h_in = max(height - 2 * t, 0.0)

    V_outer = length * width * height
    V_inner = l_in   * w_in   * h_in
    V_shell  = V_outer - V_inner

    return V_shell * inputs.foam_density


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
) -> FuselageResult:
    if inputs is None:
        inputs = FuselageInputs()
    i = inputs

    length = max(
        i.casing_factor * sizing.c_root,
        i.battery_length * i.housing_factor * i.casing_factor,
    )
    width  = i.battery_width  * i.casing_factor
    height = i.battery_height * i.casing_factor

    d_eq     = np.sqrt(width * height)
    fineness = length / d_eq
    Swet     = 2 * (length * width + length * height + width * height)
    mass     = _casing_mass(length, width, height, inputs)

    return FuselageResult(
        inputs=inputs,
        length=length,
        width=width,
        height=height,
        d_eq=d_eq,
        fineness=fineness,
        Swet=Swet,
        mass=mass,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage ---")
    print(f"  Length               : {r.length:.4f}  m")
    print(f"  Width                : {r.width:.4f}  m")
    print(f"  Height               : {r.height:.4f}  m")
    print(f"  Fineness ratio       : {r.fineness:.3f}")
    print(f"  Wetted area          : {r.Swet:.4f}  m²")
    print(f"  Casing mass          : {r.mass*1e3:.1f}  g")


# Default run at module load — exposes inputs+result fields as module attributes.
from sizing import initial_sizing as _is  # noqa: E402
_default = run(_is._default)
for _f in fields(FuselageInputs):
    globals()[_f.name] = getattr(_default.inputs, _f.name)
for _f in fields(FuselageResult):
    if _f.name != "inputs":
        globals()[_f.name] = getattr(_default, _f.name)

if __name__ == "__main__":
    summary(_default)