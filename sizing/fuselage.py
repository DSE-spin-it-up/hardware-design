from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sizing.wing import SizingResult
from structures.materials import EPP


@dataclass
class FuselageInputs:
    # Off-the-shelf battery cell envelope [m]
    battery_length: float = 0.212
    battery_width: float  = 0.090
    battery_height: float = 0.060
    # Battery aspect ratios (override fixed dims when both > 0).
    AR_lw: float = 0.0   # length / width
    AR_lh: float = 0.0   # length / height
    # Geometric margin applied uniformly to internal cross-section.
    casing_factor: float    = 1.15   # [-] all-around margin for EPP foam walls
    casing_thickness: float = 0.001  # [m] minimum hard wall thickness (fore/aft end caps)
    # How far the tube extends aft of the aileron hinge to grip the tail boom.
    tube_tail_overlap: float = 0.03  # [m]
    # EPP tearout prevention
    load_factor: float    = 1.5     # [-] battery tearout safety factor
    g: float              = 9.81    # [m/s²]
    min_foam_floor: float = 0.005   # [m] absolute minimum foam floor thickness
    pvc_snap_force: float = 372.8   # [N] dynamic peak snap force reacted by PVC support
    pvc_snap_safety_factor: float = 1.5  # [-] safety factor on PVC snap force
    min_pvc_foam_floor: float = 0.005  # [m] minimum EPP floor below PVC support
    # Raymer nose / tail fineness fractions of total aero length
    nose_fraction: float = 0.20     # [-] nose cone length / total aero length
    tail_fraction: float = 0.30     # [-] tail taper length / total aero length


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    # Outer Aerodynamic Body Dimensions
    length: float       # [m]  total aero length (nose + cylinder + tail)
    width: float        # [m]  max outer diameter in y
    height: float       # [m]  max outer diameter in z
    d_eq: float         # [m]  equivalent diameter sqrt(w*h)
    fineness: float     # [-]  length / d_eq
    Swet: float         # [m²] wetted area (Raymer cylindrical-body formula)
    # Aero sub-section lengths
    l_nose: float       # [m]
    l_cylinder: float   # [m]
    l_tail: float       # [m]
    # Internal Structural Box Dimensions
    box_length: float   # [m]
    box_width: float    # [m]
    box_height: float   # [m]
    # Battery dimensions actually used
    battery_length: float  # [m]
    battery_width: float   # [m]
    battery_height: float  # [m]
    # Derived / output fields
    volume_shell: float        = 0.0  # [m³] internal volume of aero body
    x_nose: float              = 0.0  # [m]  aero nose x from LEMAC
    battery_y_min: float       = 0.0  # [m]  lowest allowable battery bottom
    foam_floor_thickness: float = 0.0  # [m]  required foam below battery
    pvc_floor_thickness: float = 0.0  # [m]  required foam below PVC/wing tube


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
    battery_volume: float = 0.0,
    battery_x: float | None = None,
    battery_mass: float = 0.0,
    tube_back_x: float | None = None,
    tube_outer_diameter: float = 0.0,
    tube_length: float = 0.0,
) -> FuselageResult:

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

    # ------------------------------------------------------------------ 2. EPP tearout checks
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

    box_width_prelim = b_width * i.casing_factor
    pvc_bearing_area = tube_length * box_width_prelim
    pvc_design_force = i.pvc_snap_force * i.pvc_snap_safety_factor
    pvc_stress = pvc_design_force / pvc_bearing_area if pvc_bearing_area > 0 else 0.0
    pvc_floor_thickness = max(pvc_stress / epp.s_t, i.min_pvc_foam_floor)

    if pvc_stress / epp.s_t > i.min_pvc_foam_floor:
        print(f"  PVC floor: tearout governs        "
              f"(required {pvc_floor_thickness*1e3:.1f} mm for "
              f"{pvc_design_force:.1f} N design load)")
    else:
        print(f"  PVC floor: min thickness governs  "
              f"(required {pvc_floor_thickness*1e3:.1f} mm, "
              f"tearout would need {pvc_stress/epp.s_t*1e3:.1f} mm)")

    # ------------------------------------------------------------------ 3. Structural Box
    # Front wall: forward face of battery minus the end-cap thickness.
    if battery_x is not None:
        x_battery_front = battery_x - b_length / 2.0
    else:
        x_battery_front = sizing.c_root / 2.0 - b_length / 2.0

    x_nose_box = x_battery_front - i.casing_thickness

    # Back wall: rear face of PVC tube minus the end-cap thickness.
    _tube_back  = tube_back_x if tube_back_x is not None else sizing.c_root
    x_aft_box   = _tube_back + i.casing_thickness

    box_length = x_aft_box - x_nose_box
    box_width  = box_width_prelim
    tube_height = tube_outer_diameter * i.casing_factor
    battery_floor_for_height = (
        pvc_floor_thickness + 0.005
        if battery_x is not None and battery_x >= 0.0
        else foam_floor_thickness
    )
    box_height = max(
        battery_floor_for_height + b_height,
        pvc_floor_thickness + tube_height,
    )

    # ------------------------------------------------------------------ 4. Raymer-style Aero Body
    # The cylindrical midsection wraps the structural box with a 1:1 mapping —
    # outer cross-section equals box cross-section (walls already include the
    # casing_factor margin).  Nose and tail cones are added fore and aft.
    #
    #   |<-- l_nose -->|<--- l_cylinder (= box_length) --->|<-- l_tail -->|
    #
    # nose_fraction and tail_fraction are defined relative to the *total* aero length:
    #   l_total = box_length / (1 - nose_fraction - tail_fraction)

    frac_mid = 1.0 - i.nose_fraction - i.tail_fraction
    if frac_mid <= 0:
        raise ValueError(
            f"nose_fraction ({i.nose_fraction}) + tail_fraction ({i.tail_fraction}) "
            f"must be < 1.0"
        )

    l_total    = box_length / frac_mid
    l_nose     = i.nose_fraction * l_total
    l_cylinder = box_length            # structural box occupies the parallel section
    l_tail     = i.tail_fraction * l_total

    length = l_total                   # total aerodynamic length
    width  = box_width                 # max cross-section width
    height = box_height                # max cross-section height

    # Aero nose starts forward of the structural box nose by exactly l_nose.
    x_nose = x_nose_box - l_nose

    # ------------------------------------------------------------------ 5. Derived Aero Data
    d_eq     = np.sqrt(width * height)
    fineness = length / d_eq

    # Raymer wetted-area formula for a body of revolution with nose + cylinder + tail:
    #   S_wet = π * d_eq * [ 0.75 * l_nose              (ogive nose)
    #                       + l_cylinder                  (cylinder, both sides)
    #                       + 0.72 * l_tail ]             (boat-tail taper)
    # Reference: Raymer, "Aircraft Design", §12.5 component buildup.
    Swet = np.pi * d_eq * (0.75 * l_nose + l_cylinder + 0.72 * l_tail)

    # Internal volume of the cylindrical midsection (most structural material is here)
    volume_shell = np.pi * (d_eq / 2.0) ** 2 * l_cylinder

    return FuselageResult(
        inputs=inputs,
        length=length,
        width=width,
        height=height,
        d_eq=d_eq,
        fineness=fineness,
        Swet=Swet,
        l_nose=l_nose,
        l_cylinder=l_cylinder,
        l_tail=l_tail,
        box_length=box_length,
        box_width=box_width,
        box_height=box_height,
        battery_length=b_length,
        battery_width=b_width,
        battery_height=b_height,
        volume_shell=volume_shell,
        x_nose=x_nose,
        battery_y_min=battery_y_min,
        foam_floor_thickness=foam_floor_thickness,
        pvc_floor_thickness=pvc_floor_thickness,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage (Raymer Cylindrical Body) ---")
    print(f"  Battery L × W × H     : {r.battery_length:.4f} × {r.battery_width:.4f} × {r.battery_height:.4f}  m")
    print(f"  Internal Box           : {r.box_length:.4f} × {r.box_width:.4f} × {r.box_height:.4f}  m")
    print(f"  Aero body total length : {r.length:.4f}  m  "
          f"(nose {r.l_nose:.4f} + cyl {r.l_cylinder:.4f} + tail {r.l_tail:.4f})")
    print(f"  Max width × height     : {r.width:.4f} × {r.height:.4f}  m")
    print(f"  Nose x (from LEMAC)    : {r.x_nose:.4f}  m")
    print(f"  Equivalent Diameter    : {r.d_eq:.4f}  m")
    print(f"  Fineness ratio (L/D)   : {r.fineness:.3f}")
    print(f"  Wetted Area (Raymer)   : {r.Swet:.4f}  m²")
    print(f"  Cylinder internal vol  : {r.volume_shell:.6f}  m³")
    print(f"  Foam floor thickness   : {r.foam_floor_thickness*1e3:.1f}  mm")
    print(f"  PVC floor thickness    : {r.pvc_floor_thickness*1e3:.1f}  mm")


if __name__ == "__main__":
    from sizing import wing
    from sizing.wing import SizingInputs
    summary(run(wing.run(SizingInputs())))
