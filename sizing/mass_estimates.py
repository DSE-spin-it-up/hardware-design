"""Mass estimation functions for aircraft components."""

from pathlib import Path

import numpy as np

from .fuselage import run as run_fuselage, FuselageInputs
from .initial_sizing import SizingResult
from .materials import CFRP, EPP
from .propulsion_sizing import PropulsionInputs, PropulsionResult
from .structure import StructureResult
from aerodynamics.airfoil_shape import AirfoilGeometry
from .config import MaterialsConfig

CONFIG = MaterialsConfig()

def battery_mass(propulsion: PropulsionResult) -> float:
    """Battery mass [kg] from the propulsion sizing result."""
    return propulsion.battery_mass


def motor_mass(propulsion: PropulsionResult) -> float:
    """Total motor mass [kg] from the propulsion sizing result."""
    return propulsion.total_motor_mass


def prop_mass(propulsion: PropulsionResult) -> float:
    """Total propeller mass [kg] from the propulsion sizing result."""
    return propulsion.inputs.n_props * propulsion.inputs.prop_mass


def wing_mass(
    sizing: SizingResult,
    airfoil_path: str | Path,
    thickness: float = 0.1,
) -> float:
    """Estimate wing structural mass from airfoil area, span, thickness, and density.
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing wing span
    airfoil_path : str | Path
        Path to airfoil .dat file or NACA designation
    thickness : float
        Structural thickness parameter of the wing [m]
    
    Returns
    -------
    float
        Wing mass [kg]
    """
    airfoil = AirfoilGeometry(airfoil_path)
    airfoil_area = airfoil.compute_airfoil_area(chord=1.0)
    wing_volume = airfoil_area * sizing.inputs.b * thickness
    return CONFIG.wing.mass(wing_volume)


def tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
) -> float:
    """Estimate tail (horizontal + vertical stabilizer) structural mass from airfoil area, span, thickness, and density.
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing tail span `bt`
    tail_airfoil_path : str | Path
        Path to tail airfoil .dat file or NACA designation
    thickness : float
        Structural thickness parameter of the tail [m]
    
    Returns
    -------
    float
        Tail mass [kg]
    """
    airfoil = AirfoilGeometry(tail_airfoil_path)
    airfoil_area = airfoil.compute_airfoil_area(chord=1.0)
    tail_span = sizing.bt
    tail_volume = airfoil_area * tail_span * thickness
    return CONFIG.tail.mass(tail_volume)


def fuselage_mass(
    sizing: SizingResult,
    fuselage_inputs: FuselageInputs | None = None,
    airfoil_path: str | Path | None = None,
) -> float:
    """Estimate fuselage structural mass.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result
    fuselage_inputs : FuselageInputs, optional
        Fuselage design inputs
    airfoil_path : str | Path, optional
        Wing airfoil path for fuselage height sizing

    Returns
    -------
    float
        Fuselage mass [kg]
    """
    fus = run_fuselage(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    return CONFIG.fuselage.mass(fus.volume_shell)


def cg_fuselage(
    sizing: SizingResult,
    fuselage_inputs: FuselageInputs | None = None,
    airfoil_path: str | Path | None = None,
) -> float:
    """Compute fuselage CG x-position relative to LEMAC.
    
    Assumes fuselage starts at LEMAC (x = 0) and extends rearward.
    CG is at the centroid of the fuselage box.
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result
    fuselage_inputs : FuselageInputs, optional
        Fuselage design inputs
    airfoil_path : str | Path, optional
        Wing airfoil path for fuselage height sizing
    
    Returns
    -------
    float
        Fuselage CG x-position relative to LEMAC [m]
    """
    if fuselage_inputs is None:
        fuselage_inputs = FuselageInputs()
    fus = run_fuselage(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    return fus.length / 2.0


def cg_motors(sizing: SizingResult) -> float:
    """Compute motors CG x-position relative to LEMAC.
    
    Motors are mounted at the leading edge of the wing (LEMAC).
    
    Returns
    -------
    float
        Motors CG x-position relative to LEMAC [m]
    """
    return 0.0


def cg_rod_wing(airfoil_path: str | Path, sizing: SizingResult) -> float:
    """Compute first wing rod (main strut) CG x-position relative to LEMAC.
    
    Rod is positioned at the max thickness location of the wing airfoil.
    
    Parameters
    ----------
    airfoil_path : str | Path
        Wing airfoil path
    sizing : SizingResult
        Sizing result
    
    Returns
    -------
    float
        Rod CG x-position relative to LEMAC [m]
    """
    airfoil = AirfoilGeometry(airfoil_path)
    _, x_max_tc = airfoil.compute_maximum_thickness()  # fraction of chord
    return x_max_tc * sizing.c_root


def cg_rod_aileron(
    tail_airfoil_path: str | Path,
    sizing: SizingResult,
    aileron_chord_frac: float = 0.3,
) -> float:
    """Compute second rod (aileron attachment) CG x-position relative to LEMAC.
    
    Rod is positioned at the max thickness location of the aileron section.
    Aileron is typically at 70% span with 30% of wing chord.
    
    Parameters
    ----------
    tail_airfoil_path : str | Path
        Airfoil path (assuming aileron uses a section of wing airfoil or tail airfoil)
    sizing : SizingResult
        Sizing result
    aileron_chord_frac : float
        Aileron chord as fraction of local wing chord (default 0.3)
    
    Returns
    -------
    float
        Rod CG x-position relative to LEMAC [m]
    """
    airfoil = AirfoilGeometry(tail_airfoil_path)
    _, x_max_tc = airfoil.compute_maximum_thickness()
    aileron_chord = aileron_chord_frac * sizing.c_root
    return x_max_tc * aileron_chord


def cg_tail(sizing: SizingResult) -> float:
    """Compute tail CG x-position relative to LEMAC.
    
    Tail is positioned at distance L_tail behind the wing LEMAC.
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing L_tail
    
    Returns
    -------
    float
        Tail CG x-position relative to LEMAC [m]
    """
    return sizing.c_root + sizing.L_tail


def cg_tail_rod(sizing: SizingResult) -> float:
    """Compute tail rod CG x-position relative to LEMAC.

    Tail rod spans from the trailing edge of the wing root to the tail,
    so its CG is at the midpoint of that length.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing c_root and L_tail

    Returns
    -------
    float
        Tail rod CG x-position relative to LEMAC [m]
    """
    return sizing.c_root + sizing.L_tail / 2.0


def cg_wing(airfoil_path: str | Path, sizing: SizingResult) -> float:
    """Compute wing structural mass CG x-position relative to LEMAC.

    Approximated as the quarter-chord position of the root chord.

    Parameters
    ----------
    airfoil_path : str | Path
        Wing airfoil path (unused; quarter-chord is a geometry-independent approximation)
    sizing : SizingResult
        Sizing result

    Returns
    -------
    float
        Wing CG x-position relative to LEMAC [m]
    """
    return 0.25 * sizing.c_root


def cg_battery(
    rod_wing_x: float,
    rod_aileron_x: float,
) -> float:
    """Compute battery + avionics CG x-position relative to LEMAC.

    Battery is positioned midway between the two wing rods.

    Parameters
    ----------
    rod_wing_x : float
        Main wing rod CG x-position [m]
    rod_aileron_x : float
        Rear/aileron rod CG x-position [m]

    Returns
    -------
    float
        Battery CG x-position relative to LEMAC [m]
    """
    return 0.5 * (rod_wing_x + rod_aileron_x)


def cg_pvc_tubes(rod_wing_x: float, rod_aileron_x: float) -> float:
    """Compute PVC tubes CG x-position relative to LEMAC.
    
    Positioned at midpoint between the two span rods (can be refined later).
    
    Parameters
    ----------
    rod_wing_x : float
        First rod (wing) x-position [m]
    rod_aileron_x : float
        Second rod (aileron) x-position [m]
    
    Returns
    -------
    float
        PVC tubes CG x-position relative to LEMAC [m]
    """
    return (rod_wing_x + rod_aileron_x) / 2.0


def pvc_tubes_mass() -> float:
    """Get PVC tubes mass (structural connecting elements).
    
    Returns
    -------
    float
        PVC tubes mass [kg] (fixed at 225 g)
    """
    return 0.225


def compute_cg(
    sizing: SizingResult,
    propulsion: PropulsionResult,
    structure: StructureResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    battery_x: float | None = None,
    pvc_tubes_mass_override: float | None = None,
) -> dict[str, float]:
    """Compute CG x-position of all components and overall aircraft CG.
    
    All positions are relative to the leading edge of the mean aerodynamic chord (LEMAC).
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result
    propulsion : PropulsionResult
        Propulsion system result
    structure : StructureResult
        Rod sizing result; supplies wing/tail rod masses.
    airfoil_path : str | Path
        Wing airfoil path
    tail_airfoil_path : str | Path
        Tail airfoil path
    fuselage_inputs : FuselageInputs, optional
        Fuselage design inputs
    battery_x : float, optional
        Battery CG x-position; if None, defaults to midpoint between wing rods
    pvc_tubes_mass_override : float, optional
        PVC tubes mass override; if None, uses default 225 g
    
    Returns
    -------
    dict[str, float]
        Dictionary with CG x-positions [m] relative to LEMAC:
        - 'fuselage': fuselage CG
        - 'battery': battery + avionics CG
        - 'motors': motors CG
        - 'wing': wing structural CG
        - 'rod_wing': first rod (wing strut) CG
        - 'rod_aileron': second rod (aileron) CG
        - 'tail': tail CG
        - 'tail_rod': tail rod CG
        - 'pvc_tubes': PVC tubes CG
        - 'overall': weighted overall aircraft CG
    """
    # Compute individual component CGs
    x_fus = cg_fuselage(sizing, fuselage_inputs, airfoil_path)
    x_motor = cg_motors(sizing)
    x_wing = cg_wing(airfoil_path, sizing)
    x_rod_wing = cg_rod_wing(airfoil_path, sizing)
    x_rod_aileron = cg_rod_aileron(tail_airfoil_path, sizing)
    x_batt = cg_battery(x_rod_wing, x_rod_aileron) if battery_x is None else battery_x
    x_tail = cg_tail(sizing)
    x_tail_rod = cg_tail_rod(sizing)
    x_pvc = cg_pvc_tubes(x_rod_wing, x_rod_aileron)

    # Get component masses
    m_fus = fuselage_mass(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    m_batt = battery_mass(propulsion)
    m_motor = motor_mass(propulsion)
    m_wing = wing_mass(sizing, airfoil_path)
    m_rod_single = structure.mass_w
    m_tail = tail_mass(sizing, tail_airfoil_path)
    m_tail_rod = structure.mass_t
    m_pvc = pvc_tubes_mass_override if pvc_tubes_mass_override is not None else pvc_tubes_mass()

    # Compute weighted CG
    total_cg_mass = m_fus + m_batt + m_motor + m_wing + 2 * m_rod_single + m_tail + m_tail_rod + m_pvc
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
    structure: StructureResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    wing_thickness: float = 0.1,
    tail_thickness: float = 0.08,
) -> dict[str, float]:
    """Compute total aircraft mass as sum of components."""
    m_battery = battery_mass(propulsion)
    m_motors = motor_mass(propulsion)
    m_props = prop_mass(propulsion)
    m_wing = wing_mass(sizing, airfoil_path, wing_thickness)
    m_tail = tail_mass(sizing, tail_airfoil_path, tail_thickness)
    m_rod = 2 * structure.mass_w
    m_tail_rod = structure.mass_t
    m_fuselage = fuselage_mass(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    m_pvc = pvc_tubes_mass()

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