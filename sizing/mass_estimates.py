"""Mass estimation functions for aircraft components."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

from .fuselage import run as run_fuselage, FuselageInputs
from .initial_sizing import SizingResult, tail_chord
from .materials import CFRP, EPP
from .electrical_system import ElectricalResult
from .structure import d_w as rod_d, t_w as rod_t
from aerodynamics.airfoil_shape import AirfoilGeometry
from .config import MaterialsConfig

CONFIG = MaterialsConfig()

def battery_mass(electrical: ElectricalResult) -> float:
    """Battery mass [kg] from the electrical sizing result."""
    return electrical.battery_mass


def motor_mass(electrical: ElectricalResult) -> float:
    """Total motor mass [kg] from the electrical sizing result."""
    return electrical.total_motor_mass


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


def rod_mass(
    sizing: SizingResult,
    outer_diameter: float = rod_d,
    thickness: float = rod_t,
) -> float:
    """Estimate carbon-fiber rod mass using span, diameter, thickness, and material.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing wing span `b`
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
    return CONFIG.rod.mass(volume)


def tail_rod_mass(
    sizing: SizingResult,
    outer_diameter: float = rod_d,
    thickness: float = rod_t,
) -> float:
    """Estimate carbon-fiber rod mass using tail length, diameter, thickness, and material.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing tail length `L_tail`
    outer_diameter : float
        Rod outer diameter [m]
    thickness : float
        Rod wall thickness [m]

    Returns
    -------
    float
        Tail rod mass [kg]
    """
    length = sizing.L_tail
    inner_diameter = outer_diameter - 2.0 * thickness
    if inner_diameter < 0.0:
        raise ValueError("Rod thickness exceeds outer diameter")
    volume = np.pi * (outer_diameter**2 - inner_diameter**2) / 4.0 * length
    return CONFIG.rod.mass(volume)


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
    
    Motors are located at the front tip of the wing in side view,
    approximated as the wing leading edge position.
    
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
    aileron_chord_frac: float,
    sizing: SizingResult,
) -> float:
    """Compute second rod (aileron hinge) CG x-position relative to LEMAC.
    
    The rod is located at the leading edge of the aileron section defined by
    the chosen chord fraction.
    
    Parameters
    ----------
    aileron_chord_frac : float
        Aileron chord as fraction of wing root chord.
    sizing : SizingResult
        Sizing result
    
    Returns
    -------
    float
        Rod CG x-position relative to LEMAC [m]
    """
    return (1.0 - aileron_chord_frac) * sizing.c_root


def cg_tail(sizing: SizingResult) -> float:
    """Compute tail CG x-position relative to LEMAC.

    For the V-tail mass CG, use the tail root attachment point rather than
    shifting the marker rearwards by a quarter-chord offset.

    Parameters
    ----------
    sizing : SizingResult
        Sizing result containing c_root and L_tail

    Returns
    -------
    float
        Tail CG x-position relative to LEMAC [m]
    """
    return sizing.c_root + sizing.L_tail


def tail_root_x(sizing: SizingResult) -> float:
    """Tail root x-position relative to LEMAC.

    The tail rod attaches at the root of the V-tail, which is located at
    the tail arm distance behind the wing root chord.
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


def cg_tail_vertical(sizing: SizingResult) -> float:
    """Compute the V-tail mass vertical CG relative to the tail root.

    Use the vertical projected planform of the V-tail surfaces,
    based on initial sizing vertical area `Sv` and vertical span `bv`.
    """
    root_cv = tail_chord(0.0, sizing.Sv, sizing.bv, sizing.inputs.lam_t)
    tip_cv = tail_chord(0.5, sizing.Sv, sizing.bv, sizing.inputs.lam_t)
    if root_cv + tip_cv <= 0 or sizing.bv == 0.0:
        return 0.0

    return sizing.bv * (2.0 * root_cv + tip_cv) / (3.0 * (root_cv + tip_cv))


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
    electrical: ElectricalResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    battery_x: float | None = None,
    pvc_tubes_mass_override: float | None = None,
    aileron_chord_frac: float = 0.3,
) -> dict[str, float]:
    """Compute CG x-position of all components and overall aircraft CG.
    
    All positions are relative to the leading edge of the mean aerodynamic chord (LEMAC).
    
    Parameters
    ----------
    sizing : SizingResult
        Sizing result
    electrical : ElectricalResult
        Electrical system result
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
    x_rod_aileron = cg_rod_aileron(aileron_chord_frac, sizing)
    x_batt = cg_battery(x_rod_wing, x_rod_aileron) if battery_x is None else battery_x
    x_tail = cg_tail(sizing)
    x_tail_rod = cg_tail_rod(sizing)
    x_pvc = cg_pvc_tubes(x_rod_wing, x_rod_aileron)

    # Get component masses
    m_fus = fuselage_mass(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    m_batt = battery_mass(electrical)
    m_motor = motor_mass(electrical)
    m_wing = wing_mass(sizing, airfoil_path)
    m_rod_single = rod_mass(sizing)
    m_tail = tail_mass(sizing, tail_airfoil_path)
    m_tail_rod = tail_rod_mass(sizing)
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


def plot_cg_side_view(
    sizing: SizingResult,
    electrical: ElectricalResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    battery_x: float | None = None,
    pvc_tubes_mass_override: float | None = None,
    aileron_chord_frac: float = 0.3,
    show: bool = True,
) -> plt.Figure:
    """Plot a side-view CG sketch for the aircraft components.

    The fuselage is shown as a rectangle, the wing as its airfoil profile,
    the battery as a rectangle above a side-view PVC tube, and the rods
    as circles inside the cylinder. Motors, tail, and overall CG are shown
    as point masses.
    """
    if fuselage_inputs is None:
        fuselage_inputs = FuselageInputs()

    fus = run_fuselage(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    cg = compute_cg(
        sizing=sizing,
        electrical=electrical,
        airfoil_path=airfoil_path,
        tail_airfoil_path=tail_airfoil_path,
        fuselage_inputs=fuselage_inputs,
        battery_x=battery_x,
        pvc_tubes_mass_override=pvc_tubes_mass_override,
        aileron_chord_frac=aileron_chord_frac,
    )

    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    airfoil_coords = np.column_stack(
        (
            airfoil.polygon[:, 0] * root_chord,
            airfoil.polygon[:, 1] * root_chord,
        )
    )
    min_y = np.min(airfoil_coords[:, 1])
    airfoil_coords[:, 1] -= min_y
    airfoil_height = np.max(airfoil_coords[:, 1])

    x_motor = cg['motors']
    x_rod_wing = cg['rod_wing']
    x_rod_aileron = cg['rod_aileron']
    x_pvc = cg['pvc_tubes']
    x_batt = cg['battery']
    x_tail = cg['tail']
    x_tail_rod = cg['tail_rod']
    x_tail_root = tail_root_x(sizing)
    x_overall = cg['overall']

    battery_length = fus.battery_length
    battery_height = fus.battery_height
    rod_span_length = abs(x_rod_aileron - x_rod_wing)
    tube_height = max(rod_d * 1.1, 0.03)
    tube_length = max(rod_span_length + 0.1, battery_length * 0.8)
    tube_x0 = x_pvc - tube_length / 2.0
    tube_x0 = max(tube_x0, 0.0)
    # Place the PVC tube inside the wing thickness so the rods are shown
    # at their actual wing locations. The battery sits on top of the tube.
    tube_y0 = max(airfoil_height - tube_height - 0.005, 0.0)
    batt_x0 = x_batt - battery_length / 2.0
    batt_y0 = tube_y0 + tube_height + 0.005
    plot_height = max(fus.height, batt_y0 + battery_height + 0.01)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.add_patch(Rectangle((0.0, 0.0), fus.length, fus.height, fill=False, linewidth=2, label='Fuselage'))
    ax.add_patch(Polygon(airfoil_coords, closed=True, facecolor='lightblue', edgecolor='navy', alpha=0.6, label='Wing profile'))
    ax.add_patch(Rectangle((batt_x0, batt_y0), battery_length, battery_height, color='orange', alpha=0.5, label='Battery'))
    ax.add_patch(Rectangle((tube_x0, tube_y0), tube_length, tube_height, facecolor='lightgreen', alpha=0.4, edgecolor='darkgreen', label='PVC tube'))
    ax.add_patch(Ellipse((tube_x0, tube_y0 + tube_height / 2.0), tube_height, tube_height, facecolor='lightgreen', edgecolor='darkgreen', alpha=0.4))
    ax.add_patch(Ellipse((tube_x0 + tube_length, tube_y0 + tube_height / 2.0), tube_height, tube_height, facecolor='lightgreen', edgecolor='darkgreen', alpha=0.4))

    rod_radius = rod_d / 2.0
    rod_y = tube_y0 + tube_height / 2.0
    ax.add_patch(Circle((x_rod_wing, rod_y), rod_radius, color='brown', alpha=0.8, label='Wing rods'))
    ax.add_patch(Circle((x_rod_aileron, rod_y), rod_radius, color='brown', alpha=0.8))

    tail_start = x_rod_aileron + rod_radius + 0.001
    tail_end = x_tail_root
    tail_root_y = rod_y
    tail_y = tail_root_y
    plot_height = max(plot_height, tail_y + 0.05)
    ax.plot([tail_start, tail_end], [rod_y, rod_y], color='gray', linewidth=3, solid_capstyle='butt', label='Tail rod')

    le_x = np.min(airfoil_coords[:, 0])
    le_mask = np.isclose(airfoil_coords[:, 0], le_x, atol=1e-6)
    motor_y = float(np.mean(airfoil_coords[le_mask, 1])) if np.any(le_mask) else 0.5 * airfoil_height
    wing_y = 0.5 * airfoil_height
    battery_y = batt_y0 + battery_height / 2.0
    fuselage_y = fus.height / 2.0
    pvc_y = rod_y

    m_fus = fuselage_mass(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    m_batt = battery_mass(electrical)
    m_motor = motor_mass(electrical)
    m_wing = wing_mass(sizing, airfoil_path)
    m_rod_single = rod_mass(sizing)
    m_tail = tail_mass(sizing, tail_airfoil_path)
    m_tail_rod = tail_rod_mass(sizing)
    m_pvc = pvc_tubes_mass_override if pvc_tubes_mass_override is not None else pvc_tubes_mass()
    total_mass = m_fus + m_batt + m_motor + m_wing + 2 * m_rod_single + m_tail + m_tail_rod + m_pvc
    overall_y = (
        fuselage_y * m_fus
        + battery_y * m_batt
        + motor_y * m_motor
        + wing_y * m_wing
        + rod_y * 2 * m_rod_single
        + tail_y * m_tail
        + rod_y * m_tail_rod
        + pvc_y * m_pvc
    ) / total_mass if total_mass > 0 else 0.0

    ax.scatter([x_motor, x_tail, x_overall], [motor_y, tail_y, overall_y],
               color=['red', 'purple', 'black'], zorder=5)
    ax.text(x_motor, motor_y + 0.03, 'Motors', color='red', ha='center')
    ax.text(x_tail, tail_y + 0.02, 'Tail', color='purple', ha='center')
    ax.text(x_overall, overall_y + 0.02, 'Overall CG', color='black', ha='center')

    ax.set_title('Aircraft CG Side View')
    ax.set_xlabel('x [m] from LEMAC')
    ax.set_ylabel('Vertical position [m]')
    ax.set_xlim(-0.05, max(fus.length, x_tail, x_overall) + 0.2)
    ax.set_ylim(-0.05, plot_height + 0.05)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

    if show:
        plt.show()
    return fig


def total_mass(
    sizing: SizingResult,
    electrical: ElectricalResult,
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path,
    fuselage_inputs: FuselageInputs | None = None,
    wing_thickness: float = 0.1,
    tail_thickness: float = 0.08,
) -> dict[str, float]:
    """Compute total aircraft mass as sum of components."""
    m_battery = battery_mass(electrical)
    m_motors = motor_mass(electrical)
    m_wing = wing_mass(sizing, airfoil_path, wing_thickness)
    m_tail = tail_mass(sizing, tail_airfoil_path, tail_thickness)
    m_rod = 2 * rod_mass(sizing)
    m_tail_rod = tail_rod_mass(sizing)
    m_fuselage = fuselage_mass(sizing, fuselage_inputs, airfoil_path=airfoil_path)
    m_pvc = pvc_tubes_mass()

    m_total = m_battery + m_motors + m_wing + m_tail + m_rod + m_tail_rod + m_fuselage + m_pvc

    return {
        'battery': m_battery,
        'motors': m_motors,
        'wing': m_wing,
        'tail': m_tail,
        'rod': m_rod,
        'tail_rod': m_tail_rod,
        'fuselage': m_fuselage,
        'pvc_tubes': m_pvc,
        'total': m_total,
    }