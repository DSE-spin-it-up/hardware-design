"""Mass estimation functions for aircraft components."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from sizing.aileron import AileronResult
from aerodynamics.airfoil_geometry import AirfoilGeometry
from propulsion.sizing import PropulsionResult
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from structures.rods import RodResult
from weights.part_materials import PartMaterials
from structures.materials import Aluminum_6061_T6

DEFAULT_MATERIALS = PartMaterials()

SERVO_MASS = 0.052  # [kg] per servo
_PVC_SF = 1.2       # safety factor for PVC tube mass estimate


# ==============================================================================
# WIRING INPUTS & ESTIMATOR
# ==============================================================================

@dataclass
class WiringInputs:
    # Linear densities in kg/m
    rho_power_main: float = 0.0212  # 14 AWG for main wing motor lines
    rho_power_tail: float = 0.0145  # 16 AWG for the longer tail run
    rho_signal: float = 0.0012      # 28 AWG for telemetry/sensor signal lines
    
    # Assembly Factors
    bundle_overhead: float = 1.4    # 40% margin for shrink sleeves, solder, connectors
    signal_length_est: float = 2.0  # [m] baseline total length of all signal/sensor wiring


def _compute_wiring_masses(
    sizing: SizingResult, 
    inputs: WiringInputs, 
    x_batt: float, 
    x_front_motor: float, 
    x_rear_motor: float
) -> dict[str, float]:
    """Helper to calculate physical wire mass based on geometric routing distances."""
    
    # 1. Main Wing Propulsion Runs (2 Motors, Heavy Gauge)
    # Distance from battery to wing center, then spanwise out to motors
    len_wing_wire = (sizing.inputs.b / 4.0) + abs(x_front_motor - x_batt)
    m_wing_power = 2 * len_wing_wire * inputs.rho_power_main * inputs.bundle_overhead

    # 2. Tail Propulsion Motor Run (1 Motor, Medium Gauge)
    # Travels straight down the tail boom
    len_tail_wire = abs(x_rear_motor - x_batt)
    m_tail_power = len_tail_wire * inputs.rho_power_tail * inputs.bundle_overhead

    # 3. Basic Sensor & Avionics Wiring (Light Gauge)
    # Grouped as a lumped mass at the payload location
    m_signal = inputs.signal_length_est * inputs.rho_signal * inputs.bundle_overhead

    return {
        "wing_power": m_wing_power,
        "tail_power": m_tail_power,
        "signal": m_signal,
        "total": m_wing_power + m_tail_power + m_signal
    }


# ==============================================================================
# STRUCTURAL COMPONENT MASSES
# ==============================================================================

def _pvc_mass(structure: RodResult) -> float:
    """Estimate PVC tube mass from spar rod dimensions."""
    outer_d = structure.d_spar
    inner_d = structure.d_spar - structure.t_spar
    return 0.35 * np.pi * (outer_d**2 - inner_d**2) * Aluminum_6061_T6().rho * _PVC_SF


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


# ==============================================================================
# CENTER OF GRAVITY & BALANCE ENGINES
# ==============================================================================

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
    sensor_mass: float = 0.282,  # Baseline manifest mass
    wiring_inputs: WiringInputs | None = None,
) -> dict[str, float]:
    """Compute CG x-position of all components and overall aircraft CG."""
    if wiring_inputs is None:
        wiring_inputs = WiringInputs()

    x_fus = fus.x_nose + fus.length / 2.0
    x_motor = 0.0
    x_wing = 0.25 * sizing.c_root
    
    # Point mass assignment for electronics/sensors (above wing aerodynamic center)
    x_sensor = x_wing 

    x_rod_spar = _max_tc_x(airfoil_path) * sizing.c_root
    x_rod_aileron = (1.0 - aileron.inputs.c_aileron_to_c_wing) * sizing.c_root
    x_mid_rods = 0.5 * (x_rod_spar + x_rod_aileron)
    x_batt = x_mid_rods if battery_x is None else battery_x
    x_tail = 0.25 * sizing.c + sizing.lh
    x_pvc = x_mid_rods

    prop_radius = propulsion.D_prop / 2.0
    x_vt_te = x_rod_aileron + sizing.L_boom
    x_vt_le = x_vt_te - sizing.cv
    x_vt_fs = x_vt_le + 0.25 * sizing.cv
    x_ht_te = x_vt_fs - prop_radius
    x_ht_le = x_ht_te - sizing.ch
    x_tail_rod = x_rod_aileron + 0.5 * sizing.L_boom 
    x_spar_ht = x_ht_le + _max_tc_x(tail_airfoil_path) * sizing.ch
    x_control_ht = x_ht_le + (1.0 - structure.inputs.c_ruddervator_to_c_tail) * sizing.ch
    x_spar_vt = x_vt_le + _max_tc_x(tail_airfoil_path) * sizing.cv
    x_control_vt = x_vt_le + (1.0 - structure.inputs.c_ruddervator_to_c_tail) * sizing.cv

    m_fus = materials.fuselage.mass(fus.volume_shell)
    m_batt = propulsion.battery_mass
    m_motor = propulsion.total_motor_mass
    m_wing = wing_mass(sizing, airfoil_path, materials=materials)
    m_rod_spar = structure.mass_spar
    m_rod_aileron = structure.mass_aileron
    m_spar_ht = structure.mass_spar_ht
    m_control_ht = structure.mass_control_ht
    m_spar_vt = structure.mass_spar_vt
    m_control_vt = structure.mass_control_vt
    m_tail_h = hor_tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_v = ver_tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_rod = structure.mass_t
    m_pvc = _pvc_mass(structure) if pvc_tubes_mass_override is None else pvc_tubes_mass_override

    x_servo_front   = x_motor        
    x_servo_aileron = x_rod_aileron  
    x_servo_rear    = x_tail         
    x_servo_ht      = x_control_ht   
    x_servo_vt      = x_control_vt   
    x_servo_avg = (2*x_servo_front + 2*x_servo_aileron + 1*x_servo_rear + 2*x_servo_ht + 1*x_servo_vt) / 8.0
    m_servos_total = 8 * SERVO_MASS

    thickness_m = float(sizing.inputs.glass_sheet_thickness_mm) * 1e-3
    wing_sheet_area = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    m_sheet_wing = materials.sheet.mass(wing_sheet_area * thickness_m)
    m_sheet_tail_h = materials.sheet.mass(hor_tail_sheet_area * thickness_m)
    m_sheet_tail_v = materials.sheet.mass(ver_tail_sheet_area * thickness_m)
    m_glass_sheet = m_sheet_wing + m_sheet_tail_h + m_sheet_tail_v

    # Wire Masses & Routing CGs (Midpoints of cable paths)
    w_masses = _compute_wiring_masses(sizing, wiring_inputs, x_batt, x_motor, x_tail)
    x_wiring_wing = 0.5 * (x_batt + x_motor)
    x_wiring_tail = 0.5 * (x_batt + x_tail)
    x_wiring_signal = x_sensor  # Signal cables are co-located with sensor payload

    total_cg_mass = (
        m_fus + m_batt + m_motor + m_wing + m_rod_spar + m_rod_aileron
        + m_tail_h + m_tail_v + m_tail_rod + m_pvc + m_spar_ht + m_control_ht
        + m_spar_vt + m_control_vt + m_servos_total + m_glass_sheet
        + sensor_mass + w_masses['total']
    )

    if total_cg_mass > 0:
        x_cg = (
            x_fus           * m_fus
            + x_batt        * m_batt
            + x_motor       * m_motor * 2 / 3
            + x_wing        * m_wing
            + x_rod_spar    * m_rod_spar
            + x_rod_aileron * m_rod_aileron
            + x_tail        * (m_tail_h + m_tail_v + 1 / 3 * m_motor)
            + x_tail_rod    * m_tail_rod
            + x_spar_ht     * m_spar_ht
            + x_control_ht  * m_control_ht
            + x_spar_vt     * m_spar_vt
            + x_control_vt  * m_control_vt
            + x_pvc         * m_pvc
            + x_servo_avg   * m_servos_total
            + sensor_mass   * x_sensor
            + w_masses['wing_power'] * x_wiring_wing
            + w_masses['tail_power'] * x_wiring_tail
            + w_masses['signal']     * x_wiring_signal
            + ((x_wing * wing_sheet_area + x_tail * (hor_tail_sheet_area + ver_tail_sheet_area)) /
               (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) * m_glass_sheet
               if (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) > 0 else 0.0)
        ) / total_cg_mass
    else:
        x_cg = 0.0

    return {
        'fuselage':      x_fus,
        'battery':       x_batt,
        'motors':        x_motor,
        'wing':          x_wing,
        'rod_spar':      x_rod_spar,
        'rod_aileron':   x_rod_aileron,
        'tail':          x_tail,
        'tail_rod':      x_tail_rod,
        'ht_spar':       x_spar_ht,
        'ht_rud':        x_control_ht,
        'vt_spar':       x_spar_vt,
        'vt_rud':        x_control_vt,
        'pvc_tubes':     x_pvc,
        'servos':        x_servo_avg,
        'sensors':       x_sensor,
        'wiring_wing':   x_wiring_wing,
        'wiring_tail':   x_wiring_tail,
        'wiring_signal': x_wiring_signal,
        'glass_sheet':   ((x_wing * wing_sheet_area + x_tail * (hor_tail_sheet_area + ver_tail_sheet_area)) /
                          (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area)
                          if (wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area) > 0 else 0.0),
        'overall':       x_cg,
    }


def compute_y_cg(
    sizing: SizingResult,
    fus: FuselageResult,
    structure: RodResult,
    masses: dict[str, float],
    airfoil_path: str | Path,
    cg: dict[str, float],
    sensor_mass: float = 0.282,
    wiring_inputs: WiringInputs | None = None,
) -> dict[str, float]:
    """Compute vertical (y) CG position of each component and overall aircraft."""
    if wiring_inputs is None:
        wiring_inputs = WiringInputs()

    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    x_coords = airfoil.polygon[:, 0] * root_chord
    y_coords = airfoil.polygon[:, 1] * root_chord
    airfoil_y_offset = -float(np.min(y_coords))
    y_coords = y_coords + airfoil_y_offset
    airfoil_height = float(np.max(y_coords))
    
    # Point mass assignment for sensors (positioned 1cm above the upper wing skin)
    y_sensor = airfoil_height + 0.01

    tube_height = max(max(structure.d_spar, structure.d_aileron) * 1.1, 0.03)

    _, y_up_s, y_lo_s = airfoil.compute_thickness(cg["rod_spar"] / root_chord)
    _, y_up_a, y_lo_a = airfoil.compute_thickness(cg["rod_aileron"] / root_chord)
    y_rod_spar = 0.5 * (y_up_s + y_lo_s) * root_chord + airfoil_y_offset
    y_rod_aileron = 0.5 * (y_up_a + y_lo_a) * root_chord + airfoil_y_offset

    tube_y0 = 0.5 * (y_rod_spar + y_rod_aileron) - tube_height / 2.0

    le_x = float(np.min(x_coords))
    le_mask = np.isclose(x_coords, le_x, atol=1e-6)
    y_motor = (float(np.mean(y_coords[le_mask]))
               if np.any(le_mask) else 0.5 * airfoil_height)

    y_fus = fus.height / 2.0
    y_wing = 0.5 * airfoil_height
    if cg["battery"] < 0.0:
        y_batt = fus.battery_y_min + fus.battery_height / 2.0
    else:
        batt_y0 = tube_y0 + 0.005
        y_batt = batt_y0 + fus.battery_height / 2.0
        
    y_pvc = tube_y0 + tube_height / 2.0
    y_tail_rod = y_rod_aileron
    y_tail_h = y_rod_aileron
    y_ht_spar = y_rod_spar
    y_ht_elev = y_rod_spar
    y_tail_v = y_rod_aileron + 0.5 * sizing.bv
    y_vt_spar = y_rod_aileron + 0.5 * sizing.bv
    y_vt_rud = y_rod_aileron + 0.5 * sizing.bv
    y_motor_back = y_rod_aileron + sizing.bv

    y_servo_front   = y_motor        
    y_servo_aileron = y_rod_aileron  
    y_servo_rear    = y_motor_back   
    y_servo_ht      = y_ht_spar      
    y_servo_vt      = y_rod_aileron  
    y_servo_avg = (2*y_servo_front + 2*y_servo_aileron + 1*y_servo_rear + 2*y_servo_ht + 1*y_servo_vt) / 8.0

    # Wiring Layout Y CGs (Midpoints of vertical runs)
    y_wiring_wing = 0.5 * (y_batt + y_motor)
    y_wiring_tail = 0.5 * (y_batt + y_motor_back)
    y_wiring_signal = y_sensor

    m_ht_spar = masses.get("ht_spar", 0.0)
    m_ht_rud = masses.get("ht_rud", 0.0)
    m_vt_spar = masses.get("vt_spar", 0.0)
    m_vt_rud = masses.get("vt_rud", 0.0)
    m_servos = masses.get("servos", 0.0)
    
    # Extract dynamic wire mass totals safely from passed mass dictionary 
    # (Generated originally by total_mass block)
    w_masses = _compute_wiring_masses(sizing, wiring_inputs, cg['battery'], cg['motors'], cg['tail'])

    m_total = (
        masses["fuselage"] + masses["battery"] + masses["motors"]
        + masses["wing"] + masses["rod_spar"] + masses["rod_aileron"]
        + masses["hor_tail"] + masses["ver_tail"] + masses["tail_rod"] + masses["pvc_tubes"]
        + 2.0 * m_vt_spar + 2.0 * m_vt_rud
        + m_servos
        + masses.get("glass_sheet_wing", 0.0) + masses.get("glass_sheet_tail", 0.0)
        + sensor_mass + w_masses['total']
    )
    
    if m_total > 0:
        y_overall = (
            y_fus           * masses["fuselage"]
            + y_batt        * masses["battery"]
            + y_motor       * masses["motors"] * 2 / 3
            + y_wing        * masses["wing"]
            + y_rod_spar    * masses["rod_spar"]
            + y_rod_aileron * masses["rod_aileron"]
            + y_tail_h      * masses["hor_tail"]
            + y_tail_v      * masses["ver_tail"]
            + y_tail_rod    * masses["tail_rod"]
            + y_ht_spar     * m_ht_spar
            + y_ht_elev     * m_ht_rud
            + y_vt_spar     * m_vt_spar
            + y_vt_rud      * m_vt_rud
            + y_pvc         * masses["pvc_tubes"]
            + y_servo_avg   * m_servos
            + sensor_mass   * y_sensor
            + w_masses['wing_power'] * y_wiring_wing
            + w_masses['tail_power'] * y_wiring_tail
            + w_masses['signal']     * y_wiring_signal
            + y_wing        * masses.get("glass_sheet_wing", 0.0)
            + y_tail_h      * masses.get("glass_sheet_tail_h", 0.0)
            + y_tail_v      * masses.get("glass_sheet_tail_v", 0.0)
            + y_motor_back  * masses["motors"] * 1 / 3
        ) / m_total
    else:
        y_overall = 0.0

    return {
        'fuselage':           y_fus,
        'battery':            y_batt,
        'motors':             y_motor,
        'motor_back':         y_motor_back,
        'wing':               y_wing,
        'rod_spar':           y_rod_spar,
        'rod_aileron':        y_rod_aileron,
        'hor_tail':           y_tail_h,
        'ver_tail':           y_tail_v,
        'tail_rod':           y_tail_rod,
        'vt_spar':            y_vt_spar,
        'vt_rud':             y_vt_rud,
        'pvc_tubes':          y_pvc,
        'servo_front':        y_servo_front,
        'servo_aileron':      y_servo_aileron,
        'servo_rear':         y_servo_rear,
        'servo_ht':           y_servo_ht,
        'servo_vt':           y_servo_vt,
        'servos':             y_servo_avg,
        'sensors':            y_sensor,
        'glass_sheet_wing':   y_wing,
        'glass_sheet_tail_h': y_tail_h,
        'glass_sheet_tail_v': y_tail_v,
        'pvc_tube_bottom':    tube_y0,
        'overall':            y_overall,
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
    sensor_mass: float = 0.282,
    wiring_inputs: WiringInputs | None = None,
) -> dict[str, float]:
    """Compute total aircraft mass as sum of components."""
    if wiring_inputs is None:
        wiring_inputs = WiringInputs()

    m_battery = propulsion.battery_mass
    m_motors = propulsion.total_motor_mass
    m_props = propulsion.inputs.n_props * propulsion.inputs.prop_mass
    m_wing = wing_mass(sizing, airfoil_path, wing_thickness, materials=materials)
    m_tail_h = hor_tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_tail_v = ver_tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_spar_ht = structure.mass_spar_ht
    m_control_ht = structure.mass_control_ht
    m_spar_vt = structure.mass_spar_vt
    m_control_vt = structure.mass_control_vt
    m_rod_spar = structure.mass_spar
    m_rod_aileron = structure.mass_aileron
    m_tail_rod = structure.mass_t
    m_fuselage = materials.fuselage.mass(fus.volume_shell)
    m_pvc = _pvc_mass(structure)
    m_servos = 8 * SERVO_MASS
    
    # Calculate geometric wire masses based on standard component placement assumptions
    x_batt = 0.25 * sizing.c_root
    x_motor = 0.0
    x_tail = 0.25 * sizing.c + sizing.lh
    w_masses = _compute_wiring_masses(sizing, wiring_inputs, x_batt, x_motor, x_tail)
    m_wiring_total = w_masses['total']

    wing_sheet_area = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    m_sheet_wing = materials.sheet.mass(wing_sheet_area)
    m_sheet_tail_h = materials.sheet.mass(hor_tail_sheet_area)
    m_sheet_tail_v = materials.sheet.mass(ver_tail_sheet_area)
    m_glass_sheet = m_sheet_wing + m_sheet_tail_h + m_sheet_tail_v

    m_total = (
        m_battery + m_motors + m_props + m_wing + m_tail_h + m_tail_v
        + m_rod_spar + m_rod_aileron + m_spar_ht + m_control_ht
        + m_spar_vt + m_control_vt + m_tail_rod + m_fuselage + m_pvc
        + m_servos + m_glass_sheet + sensor_mass + m_wiring_total
    )
    
    return {
        'battery':            m_battery,
        'motors':             m_motors,
        'props':              m_props,
        'wing':               m_wing,
        'hor_tail':           m_tail_h,
        'ver_tail':           m_tail_v,
        'rod_spar':           m_rod_spar,
        'rod_aileron':        m_rod_aileron,
        'tail_rod':           m_tail_rod,
        'ht_spar':            m_spar_ht,
        'ht_rud':             m_control_ht,
        'vt_spar':            m_spar_vt,
        'vt_rud':             m_control_vt,
        'fuselage':           m_fuselage,
        'pvc_tubes':          m_pvc,
        'servos':             m_servos,
        'sensors':            sensor_mass,
        'wiring':             m_wiring_total,
        'glass_sheet_wing':   m_sheet_wing,
        'glass_sheet_tail_h': m_sheet_tail_h,
        'glass_sheet_tail_v': m_sheet_tail_v,
        'glass_sheet':        m_glass_sheet,
        'total':              1.21 * m_total,  # Includes standard 21% aerospace margin
    }