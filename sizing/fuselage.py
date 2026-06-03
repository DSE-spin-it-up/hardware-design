from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sizing.wing import SizingResult
from structures.materials import EPP


@dataclass
class FuselageInputs:
    # Off-the-shelf battery cell envelope [m]
    battery_length: float = 0.212
    battery_width: float = 0.090
    battery_height: float = 0.060
    # Battery aspect ratios.
    AR_lw: float = 0.0  # length / width
    AR_lh: float = 0.0  # length / height
    # Geometric margin applied uniformly to all internal components.
    casing_factor: float = 1.15  # [-] all-around margin for EPP foam walls
    casing_thickness: float = 0.001  # [m] minimum hard wall thickness
    # How far the tube extends aft of the aileron hinge to grip the tail boom.
    tube_tail_overlap: float = 0.03  # [m]
    # EPP tearout prevention
    load_factor: float = 3.0        # [-] landing impact load factor
    g: float = 9.81                 # [m/s²]
    min_foam_floor: float = 0.005   # [m] absolute minimum foam floor thickness


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    # Outer Aerodynamic Ellipsoid Dimensions
    length: float       # [m]
    width: float        # [m]
    height: float       # [m]
    d_eq: float         # equivalent diameter [m]
    fineness: float     # length / d_eq [-]
    Swet: float         # wetted area [m²]
    # Internal Structural Box Dimensions
    box_length: float   # [m]
    box_width: float    # [m]
    box_height: float   # [m]
    ellip_len: float
    ellip_hig: float
    ellip_wid: float
    # Battery dimensions
    battery_length: float  # [m] battery dimension actually used
    battery_width: float   # [m]
    battery_height: float  # [m]
    volume_shell: float = 0.0           # [m³] structural material volume of ellipsoid
    x_nose: float = 0.0                 # [m] aero fuselage nose x from LEMAC
    battery_y_min: float = 0.0          # [m] lowest allowable battery bottom from fuselage floor
    foam_floor_thickness: float = 0.0   # [m] required foam thickness below battery


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
    battery_volume: float = 0.0,
    battery_x: float | None = None,
    battery_mass: float = 0.0,
    tube_back_x: float | None = None,
    tube_outer_diameter: float = 0.0,
) -> FuselageResult:
    
    FILLED = True

    if inputs is None:
        inputs = FuselageInputs()
    i = inputs

    # ------------------------------------------------------------------ 1. Battery dims
    if i.AR_lw > 0 and i.AR_lh > 0:
        b_length = (battery_volume * i.AR_lw * i.AR_lh) ** (1 / 3)
        b_width  = b_length / i.AR_lw
        b_height = b_length / i.AR_lh
    else:
        b_length = i.battery_length
        b_width  = i.battery_width
        b_height = i.battery_height

    # ------------------------------------------------------------------ 2. EPP tearout check
    epp = EPP()
    bearing_area = b_length * b_width
    stress = (battery_mass * i.g * i.load_factor) / bearing_area if bearing_area > 0 else 0.0
    foam_floor_thickness = max(stress / epp.s_t, i.min_foam_floor)

    if stress / epp.s_t > i.min_foam_floor:
        print(f"  EPP floor: tearout governs        "
              f"(required {foam_floor_thickness*1e3:.1f} mm)")
    else:
        print(f"  EPP floor: min thickness governs  "
              f"(required {foam_floor_thickness*1e3:.1f} mm, "
              f"tearout would need {stress/epp.s_t*1e3:.1f} mm)")

    battery_y_min = foam_floor_thickness

    # ------------------------------------------------------------------ 3. Internal Box Bounds
    if battery_x is not None:
        x_battery_front = battery_x - b_length / 2.0
    else:
        x_battery_front = sizing.c_root / 2.0 - b_length / 2.0

    x_nose_box = x_battery_front - i.casing_thickness
    _tube_back = tube_back_x if tube_back_x is not None else sizing.c_root
    x_aft_box = _tube_back + i.casing_thickness

    box_length = x_aft_box - x_nose_box
    box_width = b_width * i.casing_factor
    inner_height = max(b_height, tube_outer_diameter) * i.casing_factor
    box_height = inner_height + foam_floor_thickness

    # ------------------------------------------------------------------ 4. Encompassing Ellipsoid
    # Inflate axes by sqrt(3) so the curved shell clears the rectangular box corners
    k_clearance = 0
    
    length = box_length * k_clearance
    width  = box_width * k_clearance
    height = box_height * k_clearance

    # Shift aerodynamic nose forward by the expansion delta to keep the box centered
    delta_l = (length - box_length) / 2.0
    x_nose = x_nose_box - delta_l

    # ------------------------------------------------------------------ 5. Derived Aero Data
    d_eq     = np.sqrt(width * height)
    fineness = length / d_eq
    
    # Knud Thomsen's formula for exact wetted area of an ellipsoid
    a, b, c = length / 2.0, width / 2.0, height / 2.0
    p = 1.6075
    Swet_ellipsoid = 4.0 * np.pi * (((a*b)**p + (a*c)**p + (b*c)**p) / 3.0) ** (1.0 / p)
    
    # Apply structural modifier (tail boom taper cuts off the back half of the ellipsoid)
    Swet = Swet_ellipsoid * 0.85

    if FILLED:
        volume_shell = (4.0 / 3.0) * np.pi * a * b * c
    else:
        volume_shell = 0.0

    return FuselageResult(
        inputs=inputs,
        length=length,
        width=width,
        height=height,
        d_eq=d_eq,
        fineness=fineness,
        Swet=Swet,
        box_length=box_length,
        box_width=box_width,
        box_height=box_height,
        ellip_len=a,
        ellip_hig=b,
        ellip_wid=c,
        battery_length=b_length,
        battery_width=b_width,
        battery_height=b_height,
        volume_shell=volume_shell,
        x_nose=x_nose,
        battery_y_min=battery_y_min,
        foam_floor_thickness=foam_floor_thickness,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage (Ellipsoid Model) ---")
    print(f"  Battery L × W × H    : {r.battery_length:.4f} × {r.battery_width:.4f} × {r.battery_height:.4f}  m")
    print(f"  Internal Box Bounds  : {r.box_length:.4f} × {r.box_width:.4f} × {r.box_height:.4f}  m")
    print(f"  Outer Ellipsoid Shell: {r.length:.4f} × {r.width:.4f} × {r.height:.4f}  m")
    print(f"  Nose x (from LEMAC)  : {r.x_nose:.4f}  m")
    print(f"  Equivalent Diameter  : {r.d_eq:.4f}  m")
    print(f"  ellipsoid dimensions  :{r.ellip_len:.4f} × {r.ellip_hig:.4f} × {r.ellip_wid:.4f} m")
    print(f"  Fineness ratio (L/D) : {r.fineness:.3f}")
    print(f"  Ellipsoid Wetted Area: {r.Swet:.4f}  m²")
    print(f"  Shell internal volume: {r.volume_shell:.6f}  m³")
    print(f"  Foam floor thickness : {r.foam_floor_thickness*1e3:.1f}  mm")


if __name__ == "__main__":
    from sizing import wing
    from sizing.wing import SizingInputs
    summary(run(wing.run(SizingInputs())))