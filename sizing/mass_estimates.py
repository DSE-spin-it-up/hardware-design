"""Mass estimation functions for aircraft components."""

from pathlib import Path

import numpy as np

from .electrical_system import battery_mass_from_energy, motor_mass_from_kv, ElectricalResult
from .fuselage import run as run_fuselage, FuselageInputs
from .initial_sizing import SizingResult
from .structure import rho_cfrp, d as rod_d, t as rod_t
from aerodynamics.airfoil_shape import AirfoilGeometry


def battery_mass(electrical: ElectricalResult) -> float:
    """Get battery mass from electrical system result.
    
    Parameters
    ----------
    electrical : ElectricalResult
        Result from electrical_system.run()
    
    Returns
    -------
    float
        Battery mass [kg]
    """
    return electrical.battery_mass


def motor_mass(electrical: ElectricalResult) -> float:
    """Get total motor mass from electrical system result.
    
    Parameters
    ----------
    electrical : ElectricalResult
        Result from electrical_system.run()
    
    Returns
    -------
    float
        Total mass of all motors [kg]
    """
    return electrical.total_motor_mass


def wing_mass(
    sizing: SizingResult,
    airfoil_path: str | Path,
    thickness: float = 0.1,
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
    return wing_volume * foam_density


def tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
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
    return tail_volume * foam_density


def rod_mass(
    sizing: SizingResult,
    density: float = rho_cfrp,
    outer_diameter: float = rod_d,
    thickness: float = rod_t,
) -> float:
    """Estimate carbon-fiber rod mass using span, diameter, thickness, and density.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing wing span `b`
    density : float
        Carbon fiber density [kg/m^3]
    outer_diameter : float
        Rod outer diameter [m]
    thickness : float
        Rod wall thickness [m]

    Returns
    -------
    float
        Rod mass [kg]
    """
    length = sizing.inputs.b
    inner_diameter = outer_diameter - 2.0 * thickness
    if inner_diameter < 0.0:
        raise ValueError("Rod thickness exceeds outer diameter")
    volume = np.pi * (outer_diameter**2 - inner_diameter**2) / 4.0 * length
    return volume * density


def fuselage_mass(sizing: SizingResult, fuselage_inputs: FuselageInputs | None = None) -> float:
    """Get fuselage structural mass from fuselage sizing.
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result
    fuselage_inputs : FuselageInputs, optional
        Fuselage design inputs. If None, uses defaults.
    
    Returns
    -------
    float
        Fuselage mass [kg]
    """
    fuselage_result = run_fuselage(sizing, fuselage_inputs)
    return fuselage_result.mass


def total_mass(
    sizing: SizingResult,
    electrical: ElectricalResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    wing_thickness: float = 0.1,
    tail_thickness: float = 0.08,
    foam_density: float = 48.0,
) -> dict[str, float]:
    """Compute total aircraft mass as sum of components.
    
    Parameters
    ----------
    sizing : SizingResult
        Initial sizing result
    electrical : ElectricalResult
        Electrical system result
    airfoil_path : str | Path
        Path to wing airfoil .dat file or NACA designation
    tail_airfoil_path : str | Path
        Path to tail airfoil .dat file or NACA designation
    fuselage_inputs : FuselageInputs, optional
        Fuselage design inputs
    wing_thickness : float
        Wing structural thickness [m]
    tail_thickness : float
        Tail structural thickness [m]
    foam_density : float
        Structural foam density [kg/m³]
    
    Returns
    -------
    dict[str, float]
        Dictionary with component masses and total [kg]:
        - 'battery': battery mass
        - 'motors': total motor mass
        - 'wing': wing structural mass
        - 'tail': tail structural mass
        - 'fuselage': fuselage structural mass
        - 'total': sum of all components
    """
    m_battery = battery_mass(electrical)
    m_motors = motor_mass(electrical)
    m_wing = wing_mass(sizing, airfoil_path, wing_thickness, foam_density)
    m_tail = tail_mass(sizing, tail_airfoil_path, tail_thickness, foam_density)
    m_rod = rod_mass(sizing)
    m_fuselage = fuselage_mass(sizing, fuselage_inputs)
    
    m_total = m_battery + m_motors + m_wing + m_tail + m_rod + m_fuselage
    
    return {
        'battery': m_battery,
        'motors': m_motors,
        'wing': m_wing,
        'tail': m_tail,
        'rod': m_rod,
        'fuselage': m_fuselage,
        'total': m_total,
    }