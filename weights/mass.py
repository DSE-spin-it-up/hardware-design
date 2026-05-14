"""Mass estimation functions for aircraft components."""

from pathlib import Path

from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from propulsion.sizing import PropulsionResult
from structures.rods import RodResult
from aerodynamics.airfoil_geometry import AirfoilGeometry
from weights.part_materials import PartMaterials

DEFAULT_MATERIALS = PartMaterials()

PVC_TUBES_MASS = 0.225  # [kg]
AILERON_CHORD_FRAC = 0.3


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
    x_rod_wing = _max_tc_x(airfoil_path) * sizing.c_root
    x_rod_aileron = _max_tc_x(tail_airfoil_path) * AILERON_CHORD_FRAC * sizing.c_root
    x_mid_rods = 0.5 * (x_rod_wing + x_rod_aileron)
    x_batt = x_mid_rods if battery_x is None else battery_x
    x_tail = sizing.c_root + sizing.L_tail
    x_tail_rod = sizing.c_root + sizing.L_tail / 2.0
    x_pvc = x_mid_rods

    m_fus = materials.fuselage.mass(fus.volume_shell)
    m_batt = propulsion.battery_mass
    m_motor = propulsion.total_motor_mass
    m_wing = wing_mass(sizing, airfoil_path, materials=materials)
    m_rod_single = structure.mass_w
    m_tail = tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_rod = structure.mass_t
    m_pvc = PVC_TUBES_MASS if pvc_tubes_mass_override is None else pvc_tubes_mass_override

    total_cg_mass = (
        m_fus + m_batt + m_motor + m_wing + 2 * m_rod_single
        + m_tail + m_tail_rod + m_pvc
    )
    if total_cg_mass > 0:
        x_cg = (
            x_fus * m_fus
            + x_batt * m_batt
            + x_motor * m_motor
            + x_wing * m_wing
            + x_rod_wing * m_rod_single
            + x_rod_aileron * m_rod_single
            + x_tail * m_tail
            + x_tail_rod * m_tail_rod
            + x_pvc * m_pvc
        ) / total_cg_mass
    else:
        x_cg = 0.0

    return {
        'fuselage': x_fus,
        'battery': x_batt,
        'motors': x_motor,
        'wing': x_wing,
        'rod_wing': x_rod_wing,
        'rod_aileron': x_rod_aileron,
        'tail': x_tail,
        'tail_rod': x_tail_rod,
        'pvc_tubes': x_pvc,
        'overall': x_cg,
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
    m_rod = 2 * structure.mass_w
    m_tail_rod = structure.mass_t
    m_fuselage = materials.fuselage.mass(fus.volume_shell)
    m_pvc = PVC_TUBES_MASS

    m_total = (
        m_battery + m_motors + m_props + m_wing + m_tail
        + m_rod + m_tail_rod + m_fuselage + m_pvc
    )

    return {
        'battery': m_battery,
        'motors': m_motors,
        'props': m_props,
        'wing': m_wing,
        'tail': m_tail,
        'rod': m_rod,
        'tail_rod': m_tail_rod,
        'fuselage': m_fuselage,
        'pvc_tubes': m_pvc,
        'total': m_total,
    }
