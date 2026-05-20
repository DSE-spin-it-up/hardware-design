from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sizing.wing import SizingResult
from aerodynamics.airfoil_geometry import AirfoilGeometry


@dataclass
class FuselageInputs:
    # Off-the-shelf battery cell envelope [m] — used when AR_lw == 0 or AR_lh == 0.
    battery_length: float = 0.212
    battery_width: float = 0.090
    battery_height: float = 0.060
    # Battery aspect ratios. If both > 0, dimensions are recomputed from the
    # required battery volume (from propulsion sizing) and these ratios,
    # overriding battery_length/width/height above.
    AR_lw: float = 0.0  # length / width
    AR_lh: float = 0.0  # length / height
    # Geometric margins
    housing_factor: float = 1.5  # axial battery housing factor
    casing_factor: float = 1.1   # all-around casing margin (applied to battery dims only)
    casing_thickness: float = 0.001  # [m] wall thickness of casing shell
    height_clearance: float = 0.002  # [m] fixed vertical clearance between airfoil and battery


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
    x_nose: float = 0.0  # [m] fuselage nose x from LEMAC (negative = ahead of LE)


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
    battery_volume: float = 0.0,
    airfoil_path: str | Path | None = None,
    battery_x: float | None = None,
) -> FuselageResult:
    
    FILLED = True

    if inputs is None:
        inputs = FuselageInputs()
    i = inputs

    # ------------------------------------------------------------------ battery dims
    if i.AR_lw > 0 and i.AR_lh > 0:
        # L * (L/AR_lw) * (L/AR_lh) = V  =>  L = (V * AR_lw * AR_lh)^(1/3)
        b_length = (battery_volume * i.AR_lw * i.AR_lh) ** (1 / 3)
        b_width  = b_length / i.AR_lw
        b_height = b_length / i.AR_lh
    else:
        b_length = i.battery_length
        b_width  = i.battery_width
        b_height = i.battery_height

    # ------------------------------------------------------------------ height
    # casing_factor is applied only to the battery portion.
    # The airfoil thickness is a hard geometric constraint — do not scale it.
    if airfoil_path is not None:
        airfoil = AirfoilGeometry(airfoil_path)
        airfoil_height = airfoil.global_thickness * sizing.c_root
        height = airfoil_height + b_height * i.casing_factor + i.height_clearance
    else:
        height = b_height * i.casing_factor

    # ------------------------------------------------------------------ length
    # Battery housing extent: axial housing_factor plus the two end walls.
    # casing_factor is not applied on top to avoid double-scaling the battery.
    battery_extent = b_length * i.housing_factor + 2.0 * i.casing_thickness

    # battery_x is the battery centroid measured aft-positive from the LEMAC
    # (same convention as the CG buildup). A negative value places the battery
    # ahead of the wing leading edge.
    if battery_x is not None and battery_x < 0.0:
        # Battery in front of the wing: the fuselage spans from the battery
        # housing tip (battery_x - battery_extent/2) to the wing trailing edge
        # (at c_root from the LE), so the two lengths add.
        x_nose = battery_x - battery_extent / 2.0
        length = sizing.c_root - x_nose
    else:
        # Battery nested inside the chord-length body; take whichever of the
        # root chord (with casing margin) or the battery housing is longer.
        # The body starts at the LEMAC and runs aft.
        x_nose = 0.0
        length = max(
            sizing.c_root * i.casing_factor,  # must enclose root chord
            battery_extent,                   # battery + walls only
        )

    # ------------------------------------------------------------------ width
    width = b_width * i.casing_factor

    # ------------------------------------------------------------------ derived
    d_eq     = np.sqrt(width * height)
    fineness = length / d_eq
    Swet     = 2.0 * (length * width + length * height + width * height)

    t    = i.casing_thickness
    l_in = max(length - 2.0 * t, 0.0)
    w_in = max(width  - 2.0 * t, 0.0)
    h_in = max(height - 2.0 * t, 0.0)


    if FILLED:
        volume_shell = length * width * height

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
        x_nose=x_nose,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage ---")
    print(f"  Battery L × W × H    : "
          f"{r.battery_length:.4f} × {r.battery_width:.4f} × {r.battery_height:.4f}  m")
    print(f"  Length               : {r.length:.4f}  m")
    print(f"  Nose x (from LEMAC)  : {r.x_nose:.4f}  m")
    print(f"  Width                : {r.width:.4f}  m")
    print(f"  Height               : {r.height:.4f}  m")
    print(f"  Fineness ratio       : {r.fineness:.3f}")
    print(f"  Wetted area          : {r.Swet:.4f}  m²")
    print(f"  Shell volume         : {r.volume_shell:.6f}  m³")


if __name__ == "__main__":
    from sizing import wing
    from sizing.wing import SizingInputs
    summary(run(wing.run(SizingInputs())))