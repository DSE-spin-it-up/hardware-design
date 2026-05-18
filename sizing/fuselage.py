from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np
from sizing.initial_sizing import SizingResult
from aerodynamics.airfoil_shape import AirfoilGeometry


@dataclass
class FuselageInputs:
    # Off-the-shelf battery cell envelope [m] — used when AR_lw == 0 or AR_lh == 0.
    battery_length: float = 0.212
    battery_width: float = 0.090
    battery_height: float = 0.060
    # Battery aspect ratios. If both > 0, dimensions are recomputed from the
    # required battery volume (from electrical sizing) and these ratios,
    # overriding battery_length/width/height above.
    AR_lw: float = 0.0  # length / width
    AR_lh: float = 0.0  # length / height
    # Geometric margins
    housing_factor: float = 1.5  # axial battery housing factor
    casing_factor: float = 1.1   # all-around casing margin
    casing_thickness: float = 0.001  # [m] wall thickness of casing shell


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    length: float       # [m]
    width: float        # [m]
    height: float       # [m]
    d_eq: float         # equivalent diameter [m]
    fineness: float     # length / d_eq [-]
    Swet: float         # wetted area [m^2]
    battery_length: float  # [m] battery dimension actually used
    battery_width: float   # [m]
    battery_height: float  # [m]
    volume_shell: float = 0.0  # [m³] structural material volume


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
    battery_volume: float = 0.0,
    airfoil_path: str | Path | None = None,
) -> FuselageResult:
    if inputs is None:
        inputs = FuselageInputs()
    i = inputs

    if i.AR_lw > 0 and i.AR_lh > 0:
        # L * (L/AR_lw) * (L/AR_lh) = V  =>  L = (V * AR_lw * AR_lh)^(1/3)
        b_length = (battery_volume * i.AR_lw * i.AR_lh) ** (1 / 3)
        b_width = b_length / i.AR_lw
        b_height = b_length / i.AR_lh
    else:
        b_length = i.battery_length
        b_width = i.battery_width
        b_height = i.battery_height

    if airfoil_path is not None:
        airfoil = AirfoilGeometry(airfoil_path)
        airfoil_height = airfoil.global_thickness * sizing.c_root
        # Battery sits above the PVC tube inside the fuselage, so the total
        # fuselage height must include both the wing root thickness and the
        # battery height plus a small clearance.
        height = (airfoil_height + b_height + 0.02) * i.casing_factor
    else:
        height = b_height * i.casing_factor

    length = max(
        i.casing_factor * sizing.c_root,
        b_length * i.housing_factor * i.casing_factor,
    )
    width = b_width * i.casing_factor

    d_eq     = np.sqrt(width * height)
    fineness = length / d_eq
    Swet     = 2 * (length * width + length * height + width * height)

    t = i.casing_thickness

    l_in = max(length - 2.0 * t, 0.0)
    w_in = max(width - 2.0 * t, 0.0)
    h_in = max(height - 2.0 * t, 0.0)

    volume_shell = (
        length * width * height
        - l_in * w_in * h_in
    )

    return FuselageResult(
        inputs=inputs,
        length=length,
        width=width,
        height=height,
        d_eq=d_eq,
        fineness=fineness,
        Swet=Swet,
        battery_length=b_length,
        battery_width=b_width,
        battery_height=b_height,
        volume_shell=volume_shell,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage ---")
    print(f"  Battery L × W × H    : "
          f"{r.battery_length:.4f} × {r.battery_width:.4f} × {r.battery_height:.4f}  m")
    print(f"  Length               : {r.length:.4f}  m")
    print(f"  Width                : {r.width:.4f}  m")
    print(f"  Height               : {r.height:.4f}  m")
    print(f"  Fineness ratio       : {r.fineness:.3f}")
    print(f"  Wetted area          : {r.Swet:.4f}  m²")
    print(f"  Shell volume         : {r.volume_shell:.6f}  m³")


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