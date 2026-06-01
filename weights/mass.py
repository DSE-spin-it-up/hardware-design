"""Mass estimation functions for aircraft components."""

from pathlib import Path

import numpy as np

from sizing.aileron import AileronResult
from aerodynamics.airfoil_geometry import AirfoilGeometry
from propulsion.sizing import PropulsionResult
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from structures.rods import RodResult
from weights.part_materials import PartMaterials

DEFAULT_MATERIALS = PartMaterials()

PVC_TUBES_MASS = 0.225  # [kg]


def wing_mass(
    sizing: SizingResult,
    airfoil_path: str | Path,
    thickness: float = 0.1,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    """Estimate wing structural mass from airfoil area, span, and thickness."""
    airfoil_area = AirfoilGeometry(airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.wing.mass(airfoil_area * sizing.inputs.b * thickness)


def hor_tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    """Estimate tail structural mass from airfoil area, span, and thickness."""
    airfoil_area = AirfoilGeometry(tail_airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.tail.mass(airfoil_area * sizing.bh * thickness)

def ver_tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    """Estimate tail structural mass from airfoil area, span, and thickness."""
    airfoil_area = AirfoilGeometry(tail_airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.tail.mass(airfoil_area * sizing.bv * thickness)


def _max_tc_x(airfoil_path: str | Path) -> float:
    """Chord-fraction x-location of max thickness for the given airfoil."""
    _, x_max_tc = AirfoilGeometry(airfoil_path).compute_maximum_thickness()
    return x_max_tc


def compute_cg(
    sizing: SizingResult,
    propulsion: PropulsionResult,
        structure: RodResult,
    fus: FuselageResult,
    aileron: AileronResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    battery_x: float | None = None,
    pvc_tubes_mass_override: float | None = None,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> dict[str, float]:
    """Compute CG x-position of all components and overall aircraft CG.

    All positions are relative to the leading edge of the mean aerodynamic
    chord (LEMAC).
    """
    # Fuselage centroid: nose (x_nose, negative when the body extends ahead of
    # the LE for a forward battery) plus half the length.
    x_fus = fus.x_nose + fus.length / 2.0
    
    x_motor = 0.0
    x_wing = 0.25 * sizing.c_root
    x_rod_spar = _max_tc_x(airfoil_path) * sizing.c_root
    x_rod_aileron = (1.0 - aileron.inputs.c_aileron_to_c_wing) * sizing.c_root
    x_mid_rods = 0.5 * (x_rod_spar + x_rod_aileron)
    x_batt = x_mid_rods if battery_x is None else battery_x
    # Tail mass lumped at tail AC = x_ac_wing + lh = 0.25·c + lh (from LEMAC).
    # Tail boom runs from the aileron hinge to the tail TE (length = L_boom),
    # so its centroid sits at x_rod_aileron + L_boom/2.
    x_tail = 0.25 * sizing.c + sizing.lh
    x_tail_rod = x_rod_aileron + sizing.L_boom / 2.0
    x_pvc = x_mid_rods

    # V-tail rods: tail LE from LEMAC = lh + 0.25·c − 0.25·ct, then the spar rod
    # sits at the tail airfoil's max-thickness x/c (same logic as the wing spar),
    # and the ruddervator rod sits at its hinge x/c = 1 − c_ruddervator_to_c_tail
    # (same logic as the aileron rod). Both fractions are scaled by ct.
    x_LE_tail = sizing.lh + 0.25 * sizing.c - 0.25 * max(sizing.ch, sizing.cv)
    x_spar_ht = x_LE_tail + _max_tc_x(tail_airfoil_path) * sizing.bh / sizing.inputs.ARt
    x_control_ht  = x_LE_tail + (1.0 - structure.inputs.c_ruddervator_to_c_tail) * sizing.bh / sizing.inputs.ARt
    x_spar_vt = x_LE_tail + _max_tc_x(tail_airfoil_path) * 2 * sizing.bv / sizing.inputs.ARt
    x_control_vt = x_LE_tail + (1.0 - structure.inputs.c_ruddervator_to_c_tail) * 2 * sizing.bv / sizing.inputs.ARt

    m_fus = materials.fuselage.mass(fus.volume_shell)
    m_batt = propulsion.battery_mass
    m_motor = propulsion.total_motor_mass
    m_wing = wing_mass(sizing, airfoil_path, materials=materials)
    m_rod_spar = structure.mass_spar
    m_rod_aileron = structure.mass_aileron
    # mass_vt_spar / mass_vt_rud is the mass of ONE rod; there are 2 of each
    # (one per V-tail plane), so multiply by 2 for the CG mass-weighted sum.
    m_spar_ht = structure.mass_spar_ht
    m_control_ht  = structure.mass_control_ht
    m_spar_vt = structure.mass_spar_vt
    m_control_vt = structure.mass_control_vt
    m_tail_h = hor_tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_v = ver_tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_rod = structure.mass_t
    m_pvc = PVC_TUBES_MASS if pvc_tubes_mass_override is None else pvc_tubes_mass_override

    # Glass fibre sheet mass (covering wing + tail on both sides)
    thickness_m = float(sizing.inputs.glass_sheet_thickness_mm) * 1e-3
    wing_sheet_area = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    m_sheet_wing = materials.sheet.mass(wing_sheet_area * thickness_m)
    m_sheet_tail_h = materials.sheet.mass(hor_tail_sheet_area * thickness_m)
    m_sheet_tail_v = materials.sheet.mass(ver_tail_sheet_area * thickness_m)
    m_glass_sheet = m_sheet_wing + m_sheet_tail_h + m_sheet_tail_v

    total_cg_mass = (
        m_fus + m_batt + m_motor + m_wing
        + m_rod_spar + m_rod_aileron
        + m_tail_h + m_tail_v + m_tail_rod + m_pvc
        + m_spar_ht + m_control_ht
        + m_spar_vt + m_control_vt
        + m_glass_sheet
    )
    if total_cg_mass > 0:
        x_cg = (
            x_fus        * m_fus
            + x_batt     * m_batt
            + x_motor    * m_motor*2/3
            + x_wing     * m_wing
            + x_rod_spar    * m_rod_spar
            + x_rod_aileron  * m_rod_aileron
            + x_tail     * (m_tail_h + m_tail_v +1/3*m_motor)
            + x_tail_rod * m_tail_rod
            + x_spar_ht * m_spar_ht
            + x_control_ht * m_control_ht
            + x_spar_vt * m_spar_vt
            + x_control_vt * m_control_vt
            + x_pvc      * m_pvc
            + ((x_wing * wing_sheet_area + x_tail * (hor_tail_sheet_area + ver_tail_sheet_area)) /
               (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) * m_glass_sheet if (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) > 0 else 0.0)
        ) / total_cg_mass
    else:
        x_cg = 0.0

    return {
        'fuselage':     x_fus,
        'battery':      x_batt,
        'motors':       x_motor,
        'wing':         x_wing,
        'rod_spar':     x_rod_spar,
        'rod_aileron':  x_rod_aileron,
        'tail':         x_tail,
        'tail_rod':     x_tail_rod,
        'ht_spar':      x_spar_ht,
        'ht_rud':       x_control_ht,
        'vt_spar':      x_spar_vt,
        'vt_rud':       x_control_vt,
        'pvc_tubes':    x_pvc,
        'glass_sheet':  ((x_wing * wing_sheet_area + x_tail * (hor_tail_sheet_area + ver_tail_sheet_area)) /
                         (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) if (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) > 0 else 0.0),
        'overall':      x_cg,
    }


def compute_y_cg(
    sizing: SizingResult,
    fus: FuselageResult,
    structure: RodResult,
    masses: dict[str, float],
    airfoil_path: str | Path,
    cg: dict[str, float],
) -> dict[str, float]:
    """Compute vertical (y) CG position of each component and overall aircraft.

    Vertical positions are measured from the bottom of the airfoil/fuselage,
    using the same drawing-layout conventions as the side-view sketch (tube
    sits just under the upper skin; battery sits on top of the tube; each
    rod is centred on the airfoil mid-thickness line at its own x/c).
    """
    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    x_coords = airfoil.polygon[:, 0] * root_chord
    y_coords = airfoil.polygon[:, 1] * root_chord
    airfoil_y_offset = -float(np.min(y_coords))
    y_coords = y_coords + airfoil_y_offset
    airfoil_height = float(np.max(y_coords))

    tube_height = max(max(structure.d_spar, structure.d_aileron) * 1.1, 0.03)

    # Rods sit on the airfoil mid-thickness line at their respective x/c.
    _, y_up_s, y_lo_s = airfoil.compute_thickness(cg["rod_spar"] / root_chord)
    _, y_up_a, y_lo_a = airfoil.compute_thickness(cg["rod_aileron"] / root_chord)
    y_rod_spar = 0.5 * (y_up_s + y_lo_s) * root_chord + airfoil_y_offset
    y_rod_aileron = 0.5 * (y_up_a + y_lo_a) * root_chord + airfoil_y_offset

    # Tube centred on the mean rod-centre height; battery sits on top of tube.
    tube_y0 = 0.5 * (y_rod_spar + y_rod_aileron) - tube_height / 2.0
    batt_y0 = tube_y0 + tube_height + 0.005

    le_x = float(np.min(x_coords))
    le_mask = np.isclose(x_coords, le_x, atol=1e-6)
    y_motor = (float(np.mean(y_coords[le_mask]))
               if np.any(le_mask) else 0.5 * airfoil_height)

    y_fus = fus.height / 2.0
    y_wing = 0.5 * airfoil_height
    y_batt = batt_y0 + fus.battery_height / 2.0
    y_pvc = tube_y0 + tube_height / 2.0
    y_tail_rod = y_rod_aileron
    # V-tail foam surfaces and the spar/ruddervator rods all share the same
    # panel-half-span centroid above the aileron-rod height (bv is the
    # vertical projection of one V-tail panel — see sizing.wing).
    y_tail_h = y_rod_aileron
    y_ht_spar = y_rod_spar
    y_ht_elev = y_rod_spar
    y_tail_v = y_rod_aileron + 0.5 * sizing.bv
    y_vt_spar = y_rod_aileron + 0.5 * sizing.bv
    y_vt_rud  = y_rod_aileron + 0.5 * sizing.bv
    y_motor_back = y_rod_aileron + sizing.bv
    m_ht_spar = masses.get("ht_spar", 0.0)
    m_ht_rud= masses.get("ht_rud", 0.0)
    m_vt_spar = masses.get("vt_spar", 0.0)
    m_vt_rud  = masses.get("vt_rud", 0.0)

    m_total = (
        masses["fuselage"] + masses["battery"] + masses["motors"]
        + masses["wing"] + masses["rod_spar"] + masses["rod_aileron"]
        + masses["hor_tail"] + masses["ver_tail"] + masses["tail_rod"] + masses["pvc_tubes"]
        + 2.0 * m_vt_spar + 2.0 * m_vt_rud
        + masses.get("glass_sheet_wing", 0.0) + masses.get("glass_sheet_tail", 0.0)
    )
    if m_total > 0:
        y_overall = (
            y_fus       * masses["fuselage"]
            + y_batt    * masses["battery"]
            + y_motor   * masses["motors"]*2/3
            + y_wing    * masses["wing"]
            + y_rod_spar     * masses["rod_spar"]
            + y_rod_aileron  * masses["rod_aileron"]
            + y_tail_h    * masses["hor_tail"]
            + y_tail_v    * masses["ver_tail"]
            + y_tail_rod * masses["tail_rod"]
            + y_ht_spar * m_ht_spar
            + y_ht_elev * m_ht_rud
            + y_vt_spar * m_vt_spar
            + y_vt_rud  * m_vt_rud
            + y_pvc     * masses["pvc_tubes"]
            + y_wing * masses.get("glass_sheet_wing", 0.0)
            + y_tail_h * masses.get("glass_sheet_tail_h", 0.0)
            + y_tail_v * masses.get("glass_sheet_tail_v", 0.0)
            +y_motor_back * masses["motors"]*1/3
        ) / m_total
    else:
        y_overall = 0.0

    return {
        'fuselage':    y_fus,
        'battery':     y_batt,
        'motors':      y_motor,
        'motor_back':  y_motor_back,
        'wing':        y_wing,
        'rod_spar':    y_rod_spar,
        'rod_aileron': y_rod_aileron,
        'hor_tail':    y_tail_h,
        'ver_tail':    y_tail_v,
        'tail_rod':    y_tail_rod,
        'vt_spar':     y_vt_spar,
        'vt_rud':      y_vt_rud,
        'pvc_tubes':   y_pvc,
        'glass_sheet_wing': y_wing,
        'glass_sheet_tail_h': y_tail_h,
        'glass_sheet_tail_v': y_tail_v,
        'overall':     y_overall,
    }


def total_mass(
    sizing: SizingResult,
    propulsion: PropulsionResult,
    structure: RodResult,
    fus: FuselageResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    wing_thickness: float = 0.1,
    tail_thickness: float = 0.08,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> dict[str, float]:
    """Compute total aircraft mass as sum of components."""
    m_battery = propulsion.battery_mass
    m_motors = propulsion.total_motor_mass
    m_props = propulsion.inputs.n_props * propulsion.inputs.prop_mass
    m_wing = wing_mass(sizing, airfoil_path, wing_thickness, materials=materials)
    m_tail_h = hor_tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_tail_v = ver_tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_spar_ht = structure.mass_spar_ht
    m_control_ht  = structure.mass_control_ht
    m_spar_vt = structure.mass_spar_vt
    m_control_vt = structure.mass_control_vt
    m_rod_spar    = structure.mass_spar
    m_rod_aileron = structure.mass_aileron
    m_tail_rod = structure.mass_t
    m_fuselage = materials.fuselage.mass(fus.volume_shell)
    m_pvc = PVC_TUBES_MASS

    # Glass fibre sheet mass (wing + tail, both sides)
    
    wing_sheet_area = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    m_sheet_wing = materials.sheet.mass(wing_sheet_area)
    m_sheet_tail_h = materials.sheet.mass(hor_tail_sheet_area)
    m_sheet_tail_v = materials.sheet.mass(ver_tail_sheet_area)
    m_glass_sheet = m_sheet_wing + m_sheet_tail_h + m_sheet_tail_v

    m_total = (
        m_battery + m_motors + m_props + m_wing + m_tail_h + m_tail_v
        + m_rod_spar + m_rod_aileron + m_spar_ht + m_control_ht + m_spar_vt + m_control_vt + m_tail_rod + m_fuselage + m_pvc
        + m_glass_sheet
    )
    return {
        'battery':          m_battery,
        'motors':           m_motors,
        'props':            m_props,
        'wing':             m_wing,
        'hor_tail':         m_tail_h,
        'ver_tail':         m_tail_v,
        'rod_spar':         m_rod_spar,
        'rod_aileron':      m_rod_aileron,
        'tail_rod':         m_tail_rod,
        'ht_spar':          m_spar_ht,
        'ht_rud':           m_control_ht,
        'vt_spar':          m_spar_vt,
        'vt_rud':           m_control_vt,
        'fuselage':         m_fuselage,
        'pvc_tubes':        m_pvc,
        'glass_sheet_wing': m_sheet_wing,
        'glass_sheet_tail_h': m_sheet_tail_h,
        'glass_sheet_tail_v': m_sheet_tail_v,
        'glass_sheet':      m_glass_sheet,
        'total':            m_total*1.21,
    }