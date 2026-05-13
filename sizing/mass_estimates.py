"""Mass estimation functions for aircraft components."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from .fuselage import run as run_fuselage, FuselageInputs
from .initial_sizing import SizingResult
from .materials import CFRP, EPP
from .propulsion_sizing import PropulsionResult
from .structure import d as rod_d, t as rod_t
from aerodynamics.airfoil_shape import AirfoilGeometry


def battery_mass(propulsion: PropulsionResult) -> float:
    """Battery mass [kg] from the propulsion sizing result."""
    return propulsion.battery_mass


def motor_mass(propulsion: PropulsionResult) -> float:
    """Total motor mass [kg] from the propulsion sizing result."""
    return propulsion.total_motor_mass


def wing_mass(
    sizing: SizingResult,
    airfoil_path: str | Path,
    thickness: float = 0.1,
    material: CFRP | EPP | None = None,
    foam_density: float = 48.0,
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
    foam_density : float
        Foam material density [kg/m³]
    
    Returns
    -------
    float
        Wing mass [kg]
    """
    # Load airfoil geometry and compute cross-sectional area at unit chord
    airfoil = AirfoilGeometry(airfoil_path)
    airfoil_area = airfoil.compute_airfoil_area(chord=1.0)
    
    # Volume = airfoil_area * span * thickness
    # Mass = volume * density
    wing_volume = airfoil_area * sizing.inputs.b * thickness
    if material is None:
        material = EPP()
    return material.mass(wing_volume)


def tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
    material: CFRP | EPP | None = None,
    foam_density: float = 48.0,
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
    foam_density : float
        Foam material density [kg/m³]
    
    Returns
    -------
    float
        Tail mass [kg]
    """
    # Load tail airfoil geometry and compute cross-sectional area at unit chord
    airfoil = AirfoilGeometry(tail_airfoil_path)
    airfoil_area = airfoil.compute_airfoil_area(chord=1.0)
    tail_span = sizing.bt
    
    # Volume = airfoil_area * tail_span * thickness
    # Mass = volume * density
    tail_volume = airfoil_area * tail_span * thickness
    if material is None:
        material = EPP()
    return material.mass(tail_volume)


def rod_mass(
    sizing: SizingResult,
    material: CFRP | None = None,
    outer_diameter: float = rod_d,
    thickness: float = rod_t,
) -> float:
    """Estimate carbon-fiber rod mass using span, diameter, thickness, and material.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing wing span `b`
    material : CFRP, optional
        Material used for the rod.
    outer_diameter : float
        Rod outer diameter [m]
    thickness : float
        Rod wall thickness [m]

    Returns
    -------
    float
        Rod mass [kg]
    """
    if material is None:
        material = CFRP()
    length = sizing.inputs.b
    inner_diameter = outer_diameter - 2.0 * thickness
    if inner_diameter < 0.0:
        raise ValueError("Rod thickness exceeds outer diameter")
    volume = np.pi * (outer_diameter**2 - inner_diameter**2) / 4.0 * length
    return material.mass(volume)


def tail_rod_mass(
    sizing: SizingResult,
    material: CFRP | None = None,
    outer_diameter: float = rod_d,
    thickness: float = rod_t,
) -> float:
    """Estimate carbon-fiber rod mass using tail length, diameter, thickness, and material.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing tail length `L_tail`
    material : CFRP, optional
        Material used for the tail rod.
    outer_diameter : float
        Rod outer diameter [m]
    thickness : float
        Rod wall thickness [m]

    Returns
    -------
    float
        Tail rod mass [kg]
    """
    if material is None:
        material = CFRP()
    length = sizing.L_tail
    inner_diameter = outer_diameter - 2.0 * thickness
    if inner_diameter < 0.0:
        raise ValueError("Rod thickness exceeds outer diameter")
    volume = np.pi * (outer_diameter**2 - inner_diameter**2) / 4.0 * length
    return material.mass(volume)


def fuselage_mass(
    sizing: SizingResult,
    fuselage_inputs: FuselageInputs | None = None,
    airfoil_path: str | Path | None = None,
    material: EPP | None = None,
) -> float:
    """Get fuselage structural mass from fuselage sizing.
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result
    fuselage_inputs : FuselageInputs, optional
        Fuselage design inputs. If None, uses defaults.
    airfoil_path : str | Path, optional
        Wing airfoil path for fuselage height sizing.
    material : EPP, optional
        Material used for fuselage casing.
    
    Returns
    -------
    float
        Fuselage mass [kg]
    """
    if material is None:
        material = EPP()
    if fuselage_inputs is None:
        fuselage_inputs = FuselageInputs(foam_density=material.rho)
    else:
        fuselage_inputs = replace(fuselage_inputs, foam_density=material.rho)
    fuselage_result = run_fuselage(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    return fuselage_result.mass


def total_mass(
    sizing: SizingResult,
    propulsion: PropulsionResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    wing_thickness: float = 0.1,
    tail_thickness: float = 0.08,
    foam_density: float = 48.0,
) -> dict[str, float]:
    """Compute total aircraft mass as sum of components."""
    m_battery = battery_mass(propulsion)
    m_motors = motor_mass(propulsion)
    m_wing = wing_mass(sizing, airfoil_path, wing_thickness)
    m_tail = tail_mass(sizing, tail_airfoil_path, tail_thickness)
    m_rod = rod_mass(sizing)
    m_tail_rod = tail_rod_mass(sizing)
    m_fuselage = fuselage_mass(
        sizing,
        fuselage_inputs,
        airfoil_path=airfoil_path,
        material=EPP(),
    )
    
    m_total = m_battery + m_motors + m_wing + m_tail + m_rod + m_tail_rod + m_fuselage
    
    return {
        'battery': m_battery,
        'motors': m_motors,
        'wing': m_wing,
        'tail': m_tail,
        'rod': m_rod,
        'tail_rod': m_tail_rod,
        'fuselage': m_fuselage,
        'total': m_total,
    }