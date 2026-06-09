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
    # Sleeve OD must exceed the largest attached rod OD by this factor.
    tube_clearance_factor: float = 1.10  # [-]
    # EPP tearout prevention
    load_factor: float    = 1.5     # [-] battery tearout safety factor
    g: float              = 9.81    # [m/s²]
    min_foam_floor: float = 0.005   # [m] minimum fuselage thickness below tail boom
    battery_top_cover_thickness: float = 0.005  # [m] fuselage cover above battery
    tube_snap_force: float = 372.8/2   # [N] dynamic peak snap force reacted by tube support
    tube_snap_safety_factor: float = 1.5  # [-] safety factor on tube snap force
    tube_structural_safety_factor: float = 1.5  # [-] safety factor on upward structural lift
    min_tube_foam_floor: float = 0.005  # [m] minimum EPP floor below tube support
    # Raymer nose / tail fineness fractions of total aero length
    nose_fraction: float = 0.20     # [-] nose cone length / total aero length
    tail_fraction: float = 0.30     # [-] tail taper length / total aero length
    # Bending check
    bending_safety_factor: float = 1.5  # [-] safety factor on EPP yield stress for bending


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    # Outer Aerodynamic Body Dimensions
    length: float       # [m]  total aero length (nose + cylinder + tail)
    width: float        # [m]  max outer diameter in y
    height: float       # [m]  max outer diameter in z
    d_eq: float         # [m]  circular body diameter enclosing the box
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
    structural_tube_outer_diameter: float = 0.0  # [m] sleeve OD around attached rods
    pvc_floor_thickness: float = 0.0  # [m]  required foam below structural tube
    pvc_roof_thickness: float = 0.0   # [m]  required foam above structural tube
    pvc_design_force: float = 0.0     # [N]  downward snap force used for floor sizing
    structural_lift_increment: float = 0.0  # [N] one-drone-failure structural lift increment
    structural_lift_design_force: float = 0.0  # [N] factored upward structural lift
    pvc_bearing_area: float = 0.0     # [m²] rectangular floor contact area
    pvc_lift_bearing_area: float = 0.0  # [m²] tube/rod contact area reacting structural lift
    # Bending check
    n_structural: float        = 0.0  # [-]   effective structural load factor
    bending_moment: float      = 0.0  # [N·m] peak bending moment at front spar
    bending_stress: float      = 0.0  # [Pa]  max fibre stress in EPP box section
    bending_utilisation: float = 0.0  # [-]   sigma / Y  (> 1 means failure)
    bending_required_height: float = 0.0  # [m] minimum box height from bending constraint


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
    battery_volume: float = 0.0,
    battery_x: float | None = None,
    battery_mass: float = 0.0,
    battery_bottom_y: float = 0.0,
    tube_top_y: float = 0.0,
    tube_back_x: float | None = None,
    tube_outer_diameter: float = 0.0,
    tube_length: float = 0.0,
    spar_rod_diameter: float = 0.0,
    aileron_rod_diameter: float = 0.0,
    tail_rod_diameter: float = 0.0,
    wing_section_height: float | None = None,
    structural_lift_increment: float = 0.0,
    x_front_spar: float = 0.0,        # [m] from LEMAC — max thickness x position
    n_active_drones: int = 2,          # [-] drones still flying in failure case
    m_total_failure: float = 0.0,      # [kg] total flying mass in failure case
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

    # ------------------------------------------------------------------ 2. Battery vertical placement
    # The battery sits directly on top of the tail boom. The fuselage only
    # needs to cover the battery top by the configured cover thickness.
    foam_floor_thickness = 0.0
    battery_y_min = float(battery_bottom_y)

    box_width_prelim = b_width * i.casing_factor
    tube_outer_diameter = tail_rod_diameter
    pvc_bearing_area = tube_length * box_width_prelim
    pvc_design_force = 0.0
    pvc_floor_thickness = 0.0
    exposed_rod_width = max(box_width_prelim - tube_outer_diameter, 0.0)
    pvc_lift_bearing_area = (
        tube_length * tube_outer_diameter
        + spar_rod_diameter * exposed_rod_width
        + aileron_rod_diameter * exposed_rod_width
    )
    structural_lift_design_force = 0.0
    pvc_roof_thickness = 0.0

    # ------------------------------------------------------------------ 3. Structural Box
    # Front wall: forward face of battery minus the end-cap thickness.
    if battery_x is not None:
        x_battery_front = battery_x - b_length / 2.0
    else:
        x_battery_front = sizing.c_root / 2.0 - b_length / 2.0

    x_nose_box = x_battery_front - i.casing_thickness

    # Back wall: rear face of structural tube minus the end-cap thickness.
    _tube_back  = tube_back_x if tube_back_x is not None else sizing.c_root
    x_aft_box   = _tube_back + i.casing_thickness

    box_length = x_aft_box - x_nose_box
    box_width  = box_width_prelim
    tube_height = tail_rod_diameter
    battery_top_for_height = battery_y_min + b_height + i.battery_top_cover_thickness
    available_airfoil_height = (
        float(wing_section_height)
        if wing_section_height is not None and wing_section_height > 0.0
        else 0.0
    )
    box_height = max(
        battery_top_for_height,
        float(tube_top_y),
        tube_height,
        available_airfoil_height,
    )

    # ------------------------------------------------------------------ 3b. Bending check
    # The fuselage is treated as a beam supported at the front spar.
    # The battery mass, amplified by the gust load factor, causes a bending
    # moment at that support.  The section is a hollow rectangle (outer box
    # minus battery cutout).
    #
    # Load factor: the one-drone-failure lift is evaluated at structural speed.
    # From the battery's reference frame the extra lift makes it effectively
    # heavier by:
    #   n = 1 + (n_active_drones * Delta_L_per_drone) / (m_total_failure * g)
    # The structural load_factor is then applied on top as a safety margin.

    if m_total_failure > 0.0 and x_front_spar > 0.0:
        n_structural = 1.0 + (
            n_active_drones * structural_lift_increment
        ) / (m_total_failure * i.g)
    else:
        n_structural = 1.0

    x_batt_cg = battery_x if battery_x is not None else sizing.c_root / 2.0
    x_arm = x_batt_cg - x_front_spar   # [m] positive when battery is aft of spar

    F_batt_design = 0.0
    bending_moment = F_batt_design * abs(x_arm)

    allowable_util = 1.0 / i.bending_safety_factor
    allowable_stress = 0.0

    def bending_stress_for_height(height: float) -> float:
        I_outer = (box_width * height ** 3) / 12.0
        I_inner = (b_width * b_height ** 3) / 12.0
        I_net = I_outer - I_inner
        c_bend = height / 2.0
        return (bending_moment * c_bend / I_net) if I_net > 0.0 else float("inf")

    bending_required_height = box_height
    if bending_moment > 0.0 and allowable_stress > 0.0:
        if bending_stress_for_height(box_height) > allowable_stress:
            h_lo = box_height
            h_hi = max(h_lo * 1.25, h_lo + 0.01)
            while bending_stress_for_height(h_hi) > allowable_stress:
                h_hi *= 1.5

            for _ in range(80):
                h_mid = 0.5 * (h_lo + h_hi)
                if bending_stress_for_height(h_mid) > allowable_stress:
                    h_lo = h_mid
                else:
                    h_hi = h_mid

            bending_required_height = h_hi
            box_height = max(box_height, bending_required_height)

    bending_stress = bending_stress_for_height(box_height)
    bending_utilisation = 0.0

    if False and bending_utilisation > allowable_util:
        print(f"  EPP bending: FAILS                "
              f"(σ = {bending_stress/1e3:.2f} kPa, "
              f"Y = {epp.Y/1e3:.2f} kPa, "
              f"util = {bending_utilisation:.3f}, "
              f"h_req = {bending_required_height*1e3:.1f} mm, "
              f"n_structural = {n_structural:.3f}, "
              f"x_arm = {x_arm*1e3:.1f} mm)")
    elif False:
        print(f"  EPP bending: OK                   "
              f"(σ = {bending_stress/1e3:.2f} kPa, "
              f"Y = {epp.Y/1e3:.2f} kPa, "
              f"util = {bending_utilisation:.3f}, "
              f"h_req = {bending_required_height*1e3:.1f} mm, "
              f"n_structural = {n_structural:.3f}, "
              f"x_arm = {x_arm*1e3:.1f} mm)")

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
    height = max(box_height, available_airfoil_height)

    # Aero nose starts forward of the structural box nose by exactly l_nose.
    x_nose = x_nose_box - l_nose

    # ------------------------------------------------------------------ 5. Derived Aero Data
    d_eq = np.sqrt(width**2 + height**2)
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
        structural_tube_outer_diameter=tube_outer_diameter,
        pvc_floor_thickness=pvc_floor_thickness,
        pvc_roof_thickness=pvc_roof_thickness,
        pvc_design_force=pvc_design_force,
        structural_lift_increment=structural_lift_increment,
        structural_lift_design_force=structural_lift_design_force,
        pvc_bearing_area=pvc_bearing_area,
        pvc_lift_bearing_area=pvc_lift_bearing_area,
        n_structural=n_structural,
        bending_moment=bending_moment,
        bending_stress=bending_stress,
        bending_utilisation=bending_utilisation,
        bending_required_height=bending_required_height,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage (Raymer Cylindrical Body) ---")
    print(f"  Battery L × W × H     : {r.battery_length:.4f} × {r.battery_width:.4f} × {r.battery_height:.4f}  m")
    print(f"  Internal box length    : {r.box_length:.4f}  m")
    print(f"  Aero body total length : {r.length:.4f}  m  "
          f"(nose {r.l_nose:.4f} + cyl {r.l_cylinder:.4f} + tail {r.l_tail:.4f})")
    print(f"  Box width x height     : {r.width:.4f} x {r.height:.4f}  m")
    print(f"  Surrounding diameter   : {r.d_eq:.4f}  m")
    print(f"  Nose x (from LEMAC)    : {r.x_nose:.4f}  m")
    print(f"  Fineness ratio (L/D)   : {r.fineness:.3f}")
    print(f"  Wetted Area (Raymer)   : {r.Swet:.4f}  m²")
    print(f"  Cylinder internal vol  : {r.volume_shell:.6f}  m³")
    print(f"  Battery bottom y       : {r.battery_y_min:.4f}  m")
    print(f"  Battery top cover      : {r.inputs.battery_top_cover_thickness*1e3:.1f}  mm")
    print(f"  Tail boom OD           : {r.structural_tube_outer_diameter*1e3:.1f}  mm")
    print(f"  Tube tearout sizing    : disabled")
    print(f"  Structural lift inc.   : {r.structural_lift_increment:.1f}  N")
    print(f"  Tube bearing area      : {r.pvc_bearing_area:.6f}  m²")
    print(f"  Tube/rod contact area  : {r.pvc_lift_bearing_area:.6f}  m²")
    print(f"  Battery bending sizing : disabled")


if __name__ == "__main__":
    from sizing import wing
    from sizing.wing import SizingInputs
    summary(run(wing.run(SizingInputs())))
