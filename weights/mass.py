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


def tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    """Estimate tail structural mass from airfoil area, span, and thickness."""
    airfoil_area = AirfoilGeometry(tail_airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.tail.mass(airfoil_area * sizing.bt * thickness)


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
    x_fus = fus.length / 2.0
    x_motor = 0.0
    x_wing = 0.25 * sizing.c_root
    x_rod_spar = _max_tc_x(airfoil_path) * sizing.c_root
    x_rod_aileron = (1.0 - aileron.inputs.c_aileron_to_c_wing) * sizing.c_root
    x_mid_rods = 0.5 * (x_rod_spar + x_rod_aileron)
    x_batt = x_mid_rods if battery_x is None else battery_x
    x_tail = sizing.c_root + sizing.L_tail
    x_tail_rod = sizing.c_root + sizing.L_tail / 2.0
    x_pvc = x_mid_rods

    m_fus = materials.fuselage.mass(fus.volume_shell)
    m_batt = propulsion.battery_mass
    m_motor = propulsion.total_motor_mass
    m_wing = wing_mass(sizing, airfoil_path, materials=materials)
    m_rod_spar = structure.mass_spar        
    m_rod_aileron = structure.mass_aileron  
    m_vt_spar = structure.mass_vt_spar      # 2 planes
    m_vt_rud  = structure.mass_vt_rud       # 2 planes
    m_tail = tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_rod = structure.mass_t
    m_pvc = PVC_TUBES_MASS if pvc_tubes_mass_override is None else pvc_tubes_mass_override

    # Glass fibre sheet mass (covering wing + tail on both sides)
    thickness_m = float(sizing.inputs.glass_sheet_thickness_mm) * 1e-3
    wing_sheet_area = 2.0 * sizing.Sw
    tail_sheet_area = 2.0 * sizing.St
    m_sheet_wing = materials.sheet.mass(wing_sheet_area * thickness_m)
    m_sheet_tail = materials.sheet.mass(tail_sheet_area * thickness_m)
    m_glass_sheet = m_sheet_wing + m_sheet_tail

    total_cg_mass = (
        m_fus + m_batt + m_motor + m_wing
        + m_rod_spar + m_rod_aileron
        + m_tail + m_tail_rod + m_pvc
        + m_glass_sheet
    )
    if total_cg_mass > 0:
        x_cg = (
            x_fus        * m_fus
            + x_batt     * m_batt
            + x_motor    * m_motor
            + x_wing     * m_wing
            + x_rod_spar    * m_rod_spar
            + x_rod_aileron  * m_rod_aileron
            + x_tail     * m_tail
            + x_tail_rod * m_tail_rod
            + x_pvc      * m_pvc
            + ((x_wing * wing_sheet_area + x_tail * tail_sheet_area) /
               (wing_sheet_area + tail_sheet_area) * m_glass_sheet if (wing_sheet_area + tail_sheet_area) > 0 else 0.0)
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
        'rod_tail_spar':    m_vt_spar,
        'rod_tail_rud':     m_vt_rud,
        'pvc_tubes':    x_pvc,
        'glass_sheet':  ((x_wing * wing_sheet_area + x_tail * tail_sheet_area) /
                         (wing_sheet_area + tail_sheet_area) if (wing_sheet_area + tail_sheet_area) > 0 else 0.0),
        'overall':      x_cg,
    }


def compute_y_cg(
    sizing: SizingResult,
    fus: FuselageResult,
    structure: RodResult,
    masses: dict[str, float],
    airfoil_path: str | Path,
) -> dict[str, float]:
    """Compute vertical (y) CG position of each component and overall aircraft.

    Vertical positions are measured from the bottom of the airfoil/fuselage,
    using the same drawing-layout conventions as the side-view sketch (tube
    sits just under the upper skin; battery sits on top of the tube; rods
    pass through the tube centre).
    """
    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    x_coords = airfoil.polygon[:, 0] * root_chord
    y_coords = airfoil.polygon[:, 1] * root_chord
    y_coords = y_coords - float(np.min(y_coords))
    airfoil_height = float(np.max(y_coords))

    tube_height = max(max(structure.d_spar, structure.d_aileron) * 1.1, 0.03)
    tube_y0 = max(airfoil_height - tube_height - 0.005, 0.0)
    batt_y0 = tube_y0 + tube_height + 0.005
    rod_y = tube_y0 + tube_height / 2.0

    le_x = float(np.min(x_coords))
    le_mask = np.isclose(x_coords, le_x, atol=1e-6)
    y_motor = (float(np.mean(y_coords[le_mask]))
               if np.any(le_mask) else 0.5 * airfoil_height)

    y_fus = fus.height / 2.0
    y_wing = 0.5 * airfoil_height
    y_batt = batt_y0 + fus.battery_height / 2.0
    y_pvc = rod_y
    y_tail = rod_y
    y_tail_rod = rod_y

    m_total = (
        masses["fuselage"] + masses["battery"] + masses["motors"]
        + masses["wing"] + masses["rod_spar"] + masses["rod_aileron"]
        + masses["tail"] + masses["tail_rod"] + masses["pvc_tubes"]
        + masses.get("glass_sheet_wing", 0.0) + masses.get("glass_sheet_tail", 0.0)
    )
    if m_total > 0:
        y_overall = (
            y_fus       * masses["fuselage"]
            + y_batt    * masses["battery"]
            + y_motor   * masses["motors"]
            + y_wing    * masses["wing"]
            + rod_y     * masses["rod_spar"]
            + rod_y     * masses["rod_aileron"]
            + y_tail    * masses["tail"]
            + y_tail_rod * masses["tail_rod"]
            + y_pvc     * masses["pvc_tubes"]
            + y_wing * masses.get("glass_sheet_wing", 0.0)
            + y_tail * masses.get("glass_sheet_tail", 0.0)
        ) / m_total
    else:
        y_overall = 0.0

    return {
        'fuselage':    y_fus,
        'battery':     y_batt,
        'motors':      y_motor,
        'wing':        y_wing,
        'rod_spar':    rod_y,
        'rod_aileron': rod_y,
        'tail':        y_tail,
        'tail_rod':    y_tail_rod,
        'pvc_tubes':   y_pvc,
        'glass_sheet_wing': y_wing,
        'glass_sheet_tail': y_tail,
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
    m_tail = tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_vt_spar = structure.mass_vt_spar
    m_vt_rud  = structure.mass_vt_rud
    m_rod_spar    = structure.mass_spar
    m_rod_aileron = structure.mass_aileron
    m_tail_rod = structure.mass_t
    m_fuselage = materials.fuselage.mass(fus.volume_shell)
    m_pvc = PVC_TUBES_MASS

    # Glass fibre sheet mass (wing + tail, both sides)
    
    wing_sheet_area = 2.0 * sizing.Sw
    tail_sheet_area = 2.0 * sizing.St
    m_sheet_wing = materials.sheet.mass(wing_sheet_area)
    m_sheet_tail = materials.sheet.mass(tail_sheet_area)
    m_glass_sheet = m_sheet_wing + m_sheet_tail

    m_total = (
        m_battery + m_motors + m_props + m_wing + m_tail
        + m_rod_spar + m_rod_aileron + 2* m_vt_spar + 2*m_vt_rud + m_tail_rod + m_fuselage + m_pvc
        + m_glass_sheet
    )
    return {
        'battery':          m_battery,
        'motors':           m_motors,
        'props':            m_props,
        'wing':             m_wing,
        'tail':             m_tail,
        'rod_spar':         m_rod_spar,
        'rod_aileron':      m_rod_aileron,
        'tail_rod':         m_tail_rod,
        'vt_spar':          m_vt_spar,
        'vt_rud':           m_vt_rud,
        'fuselage':         m_fuselage,
        'pvc_tubes':        m_pvc,
        'glass_sheet_wing': m_sheet_wing,
        'glass_sheet_tail': m_sheet_tail,
        'glass_sheet':      m_glass_sheet,
        'total':            m_total*1.4,
    }