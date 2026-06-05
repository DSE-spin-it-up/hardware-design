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
    rho_power_main: float = 0.0212
    rho_power_tail: float = 0.0145
    rho_signal: float = 0.0012
    bundle_overhead: float = 1.4
    signal_length_est: float = 2.0


def _compute_wiring_masses(
    sizing: SizingResult,
    inputs: WiringInputs,
    x_batt: float,
    x_front_motor: float,
    x_rear_motor: float,
) -> dict[str, float]:
    len_wing_wire = (sizing.inputs.b / 4.0) + abs(x_front_motor - x_batt)
    m_wing_power = 2 * len_wing_wire * inputs.rho_power_main * inputs.bundle_overhead

    len_tail_wire = abs(x_rear_motor - x_batt)
    m_tail_power = len_tail_wire * inputs.rho_power_tail * inputs.bundle_overhead

    m_signal = inputs.signal_length_est * inputs.rho_signal * inputs.bundle_overhead

    return {
        "wing_power": m_wing_power,
        "tail_power": m_tail_power,
        "signal": m_signal,
        "total": m_wing_power + m_tail_power + m_signal,
    }


# ==============================================================================
# STRUCTURAL HELPERS
# ==============================================================================

def _fuselage_x_centroid(fus: FuselageResult) -> float:
    box_x0 = fus.x_nose + fus.l_nose
    box_x1 = box_x0 + fus.l_cylinder

    nose_area     = fus.l_nose     / 2.0
    cylinder_area = fus.l_cylinder
    tail_area     = fus.l_tail     / 2.0
    total_area    = nose_area + cylinder_area + tail_area
    if total_area <= 0.0:
        return fus.x_nose + fus.length / 2.0

    return (
        (box_x0 - fus.l_nose / 3.0)  * nose_area
        + (0.5 * (box_x0 + box_x1))  * cylinder_area
        + (box_x1 + fus.l_tail / 3.0) * tail_area
    ) / total_area


def _tube_y_bounds(
    structure: RodResult,
    fus: FuselageResult,
    y_rod_spar: float,
    y_rod_aileron: float,
) -> tuple[float, float]:
    max_d  = max(structure.d_spar, structure.d_aileron)
    margin = 0.5 * max(fus.inputs.casing_factor - 1.0, 0.0) * max_d
    y0 = min(
        y_rod_spar    - structure.d_spar    / 2.0,
        y_rod_aileron - structure.d_aileron / 2.0,
    ) - margin
    y1 = max(
        y_rod_spar    + structure.d_spar    / 2.0,
        y_rod_aileron + structure.d_aileron / 2.0,
    ) + margin
    return y0, y1


def _pvc_mass(structure: RodResult, tube_length: float) -> float:
    outer_d = structure.d_spar
    inner_d = structure.d_spar - structure.t_spar
    return tube_length * np.pi * (outer_d**2 - inner_d**2) * Aluminum_6061_T6().rho * _PVC_SF


# ==============================================================================
# SURFACE MASSES
# ==============================================================================

def wing_mass(
    sizing: SizingResult,
    airfoil_path: str | Path,
    thickness: float = 0.1,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    airfoil_area = AirfoilGeometry(airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.wing.mass(airfoil_area * sizing.inputs.b * thickness)


def hor_tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    airfoil_area = AirfoilGeometry(tail_airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.tail.mass(airfoil_area * sizing.bh * thickness)


def ver_tail_mass(
    sizing: SizingResult,
    tail_airfoil_path: str | Path,
    thickness: float = 0.08,
    materials: PartMaterials = DEFAULT_MATERIALS,
) -> float:
    airfoil_area = AirfoilGeometry(tail_airfoil_path).compute_airfoil_area(chord=1.0)
    return materials.tail.mass(airfoil_area * sizing.bv * thickness)


def _max_tc_x(airfoil_path: str | Path) -> float:
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
    sensor_mass: float = 0.282,
    wiring_inputs: WiringInputs | None = None,
) -> dict[str, float]:
    """Compute CG x-position of all components and overall aircraft CG.

    All positions are relative to the leading edge of the mean aerodynamic
    chord (LEMAC).
    """
    if wiring_inputs is None:
        wiring_inputs = WiringInputs()

    x_fus = _fuselage_x_centroid(fus)

    x_motor_front = 0.0
    x_wing        = 0.25 * sizing.c_root
    x_sensor      = x_wing

    x_rod_spar    = _max_tc_x(airfoil_path) * sizing.c_root
    x_rod_aileron = (1.0 - aileron.inputs.c_aileron_to_c_wing) * sizing.c_root
    x_mid_rods    = 0.5 * (x_rod_spar + x_rod_aileron)
    x_batt        = x_mid_rods if battery_x is None else battery_x
    x_tail        = 0.25 * sizing.c + sizing.lh
    tube_x0       = x_rod_spar    - fus.inputs.tube_tail_overlap
    tube_x1       = x_rod_aileron + fus.inputs.tube_tail_overlap
    tube_length   = tube_x1 - tube_x0
    x_pvc         = 0.5 * (tube_x0 + tube_x1)

    prop_radius  = propulsion.D_prop / 2.0
    x_vt_te      = x_rod_aileron + sizing.L_boom
    x_vt_le      = x_vt_te - sizing.cv
    x_vt_fs      = x_vt_le + 0.25 * sizing.cv
    x_motor_back = x_vt_fs
    x_ht_te      = x_vt_fs - prop_radius
    x_ht_le      = x_ht_te - sizing.ch
    x_tail_rod   = x_rod_aileron + 0.5 * sizing.L_boom
    x_spar_ht    = x_ht_le + _max_tc_x(tail_airfoil_path) * sizing.ch
    x_control_ht = x_ht_le + (1.0 - structure.inputs.c_ruddervator_to_c_tail) * sizing.ch
    x_spar_vt    = x_vt_le + _max_tc_x(tail_airfoil_path) * sizing.cv
    x_control_vt = x_vt_le + (1.0 - structure.inputs.c_ruddervator_to_c_tail) * sizing.cv

    m_fus        = materials.fuselage.mass(fus.volume_shell)
    m_batt       = propulsion.battery_mass
    m_motor      = propulsion.total_motor_mass
    m_wing       = wing_mass(sizing, airfoil_path, materials=materials)
    m_rod_spar   = structure.mass_spar
    m_rod_aileron= structure.mass_aileron
    m_spar_ht    = structure.mass_spar_ht
    m_control_ht = structure.mass_control_ht
    m_spar_vt    = structure.mass_spar_vt
    m_control_vt = structure.mass_control_vt
    m_tail_h     = hor_tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_v     = ver_tail_mass(sizing, tail_airfoil_path, materials=materials)
    m_tail_rod   = structure.mass_t
    m_pvc        = (_pvc_mass(structure, tube_length)
                    if pvc_tubes_mass_override is None else pvc_tubes_mass_override)

    x_servo_front   = x_motor_front
    x_servo_aileron = x_rod_aileron
    x_servo_rear    = x_motor_back
    x_servo_ht      = x_control_ht
    x_servo_vt      = x_control_vt
    x_servo_avg = (
        2 * x_servo_front
        + 2 * x_servo_aileron
        + 1 * x_servo_rear
        + 2 * x_servo_ht
        + 1 * x_servo_vt
    ) / 8.0
    m_servos_total = 8 * SERVO_MASS

    thickness_m         = float(sizing.inputs.glass_sheet_thickness_mm) * 1e-3
    wing_sheet_area     = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    m_sheet_wing        = materials.sheet.mass(wing_sheet_area     * thickness_m)
    m_sheet_tail_h      = materials.sheet.mass(hor_tail_sheet_area * thickness_m)
    m_sheet_tail_v      = materials.sheet.mass(ver_tail_sheet_area * thickness_m)
    m_glass_sheet       = m_sheet_wing + m_sheet_tail_h + m_sheet_tail_v

    w_masses       = _compute_wiring_masses(sizing, wiring_inputs, x_batt, x_motor_front, x_motor_back)
    x_wiring_wing  = 0.5 * (x_batt + x_motor_front)
    x_wiring_tail  = 0.5 * (x_batt + x_motor_back)
    x_wiring_signal = x_sensor

    total_sheet_area = wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area
    x_glass_sheet = (
        (x_wing * wing_sheet_area + x_tail * (hor_tail_sheet_area + ver_tail_sheet_area))
        / total_sheet_area
        if total_sheet_area > 0 else 0.0
    )

    total_cg_mass = (
        m_fus + m_batt + m_motor + m_wing
        + m_rod_spar + m_rod_aileron
        + m_tail_h + m_tail_v + m_tail_rod + m_pvc
        + m_spar_ht + m_control_ht
        + m_spar_vt + m_control_vt
        + m_servos_total
        + m_glass_sheet
        + sensor_mass + w_masses['total']
    )
    if total_cg_mass > 0:
        x_cg = (
            x_fus            * m_fus
            + x_batt         * m_batt
            + x_motor_front  * m_motor * 2 / 3
            + x_motor_back   * m_motor * 1 / 3
            + x_wing         * m_wing
            + x_rod_spar     * m_rod_spar
            + x_rod_aileron  * m_rod_aileron
            + x_tail         * (m_tail_h + m_tail_v)
            + x_tail_rod     * m_tail_rod
            + x_spar_ht      * m_spar_ht
            + x_control_ht   * m_control_ht
            + x_spar_vt      * m_spar_vt
            + x_control_vt   * m_control_vt
            + x_pvc          * m_pvc
            + x_servo_avg    * m_servos_total
            + x_glass_sheet  * m_glass_sheet
            + sensor_mass    * x_sensor
            + w_masses['wing_power'] * x_wiring_wing
            + w_masses['tail_power'] * x_wiring_tail
            + w_masses['signal']     * x_wiring_signal
        ) / total_cg_mass
    else:
        x_cg = 0.0

    return {
        'fuselage':       x_fus,
        'battery':        x_batt,
        'motors':         x_motor_front,
        'motor_back':     x_motor_back,
        'wing':           x_wing,
        'rod_spar':       x_rod_spar,
        'rod_aileron':    x_rod_aileron,
        'tail':           x_tail,
        'tail_rod':       x_tail_rod,
        'ht_spar':        x_spar_ht,
        'ht_rud':         x_control_ht,
        'vt_spar':        x_spar_vt,
        'vt_rud':         x_control_vt,
        'pvc_tubes':      x_pvc,
        'servos':         x_servo_avg,
        'sensors':        x_sensor,
        'wiring_wing':    x_wiring_wing,
        'wiring_tail':    x_wiring_tail,
        'wiring_signal':  x_wiring_signal,
        'glass_sheet':    x_glass_sheet,
        'overall':        x_cg,
    }


def compute_y_cg(
    sizing: SizingResult,
    fus: FuselageResult,
    structure: RodResult,
    masses: dict[str, float],
    airfoil_path: str | Path,
    cg: dict[str, float],
    battery_y0: float | None = None,
    sensor_mass: float = 0.282,
    wiring_inputs: WiringInputs | None = None,
) -> dict[str, float]:
    """Compute vertical (y) CG position of each component and overall aircraft.

    Vertical positions are measured from the bottom of the airfoil/fuselage.

    Also computes y_ac: the vertical position of the wing aerodynamic centre,
    defined as the camber-line height at x/c = 0.25 of the root airfoil.
    This is the correct reference point for thrust pitching moment arms in the
    scissor trim balance — all moments in the balance are about the AC, so
    Z_T = y_motor - y_ac (not y_motor - y_cg_overall).
    """
    if wiring_inputs is None:
        wiring_inputs = WiringInputs()

    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    x_coords = airfoil.polygon[:, 0] * root_chord
    y_coords = airfoil.polygon[:, 1] * root_chord
    airfoil_y_offset = -float(np.min(y_coords))
    y_coords = y_coords + airfoil_y_offset
    airfoil_height = float(np.max(y_coords))

    # Rods sit on the airfoil mid-thickness line at their respective x/c.
    _, y_up_s, y_lo_s = airfoil.compute_thickness(cg["rod_spar"]    / root_chord)
    _, y_up_a, y_lo_a = airfoil.compute_thickness(cg["rod_aileron"] / root_chord)
    y_rod_spar    = 0.5 * (y_up_s + y_lo_s) * root_chord + airfoil_y_offset
    y_rod_aileron = 0.5 * (y_up_a + y_lo_a) * root_chord + airfoil_y_offset

    # Wing AC vertical position: camber-line height at x/c = 0.25.
    # This is the correct moment reference for the scissor trim balance —
    # all moment contributions (wing Cm_ac, tail, thrust) must be about
    # the same point, which is the aerodynamic centre.
    _, y_up_ac, y_lo_ac = airfoil.compute_thickness(0.25)
    y_ac_pre_shift = 0.5 * (y_up_ac + y_lo_ac) * root_chord + airfoil_y_offset

    # Tube wraps both wing rods with the configured casing margin.
    tube_y0, tube_y1 = _tube_y_bounds(structure, fus, y_rod_spar, y_rod_aileron)
    wing_y_shift = fus.pvc_floor_thickness - tube_y0
    y_coords      += wing_y_shift
    y_rod_spar    += wing_y_shift
    y_rod_aileron += wing_y_shift
    tube_y0       += wing_y_shift
    tube_y1       += wing_y_shift
    tube_height    = tube_y1 - tube_y0

    # Apply the same shift to the AC vertical position.
    y_ac = y_ac_pre_shift + wing_y_shift

    le_x    = float(np.min(x_coords))
    le_mask = np.isclose(x_coords, le_x, atol=1e-6)
    y_motor = (float(np.mean(y_coords[le_mask]))
               if np.any(le_mask) else 0.5 * airfoil_height)

    if battery_y0 is not None:
        batt_y0 = battery_y0
    elif cg["battery"] < 0.0:
        batt_y0 = fus.battery_y_min
    else:
        batt_y0 = tube_y0 + 0.005
    box_y0 = min(
        batt_y0 - fus.foam_floor_thickness,
        tube_y0 - fus.pvc_floor_thickness,
    )
    box_yc = box_y0 + fus.box_height / 2.0

    y_fus      = box_yc
    y_sensor   = box_yc
    y_wing     = wing_y_shift + 0.5 * airfoil_height
    y_batt     = batt_y0 + fus.battery_height / 2.0
    y_pvc      = tube_y0 + tube_height / 2.0
    y_tail_rod = y_rod_aileron
    y_tail_h   = y_rod_aileron
    y_ht_spar  = y_rod_aileron
    y_ht_elev  = y_rod_aileron
    y_tail_v   = y_rod_aileron + 0.5 * sizing.bv
    y_vt_spar  = y_rod_aileron + 0.5 * sizing.bv
    y_vt_rud   = y_rod_aileron + 0.5 * sizing.bv
    y_motor_back = y_rod_aileron + sizing.bv

    y_servo_front   = y_motor
    y_servo_aileron = y_rod_aileron
    y_servo_rear    = y_motor_back
    y_servo_ht      = y_ht_spar
    y_servo_vt      = y_rod_aileron
    y_servo_avg = (
        2 * y_servo_front
        + 2 * y_servo_aileron
        + 1 * y_servo_rear
        + 2 * y_servo_ht
        + 1 * y_servo_vt
    ) / 8.0

    y_wiring_wing   = y_pvc
    y_wiring_tail   = y_pvc
    y_wiring_signal = box_yc

    m_ht_spar = masses.get("ht_spar", 0.0)
    m_ht_rud  = masses.get("ht_rud",  0.0)
    m_vt_spar = masses.get("vt_spar", 0.0)
    m_vt_rud  = masses.get("vt_rud",  0.0)
    m_servos  = masses.get("servos",  0.0)

    w_masses = _compute_wiring_masses(
        sizing, wiring_inputs, cg['battery'], cg['motors'], cg['tail']
    )

    m_total = (
        masses["fuselage"] + masses["battery"] + masses["motors"]
        + masses["wing"] + masses["rod_spar"] + masses["rod_aileron"]
        + masses["hor_tail"] + masses["ver_tail"] + masses["tail_rod"] + masses["pvc_tubes"]
        + m_ht_spar + m_ht_rud
        + m_vt_spar + m_vt_rud
        + m_servos
        + masses.get("glass_sheet_wing",   0.0)
        + masses.get("glass_sheet_tail_h", 0.0)
        + masses.get("glass_sheet_tail_v", 0.0)
        + sensor_mass + w_masses['total']
    )
    if m_total > 0:
        y_overall = (
            y_fus            * masses["fuselage"]
            + y_batt         * masses["battery"]
            + y_motor        * masses["motors"] * 2 / 3
            + y_motor_back   * masses["motors"] * 1 / 3
            + y_wing         * masses["wing"]
            + y_rod_spar     * masses["rod_spar"]
            + y_rod_aileron  * masses["rod_aileron"]
            + y_tail_h       * masses["hor_tail"]
            + y_tail_v       * masses["ver_tail"]
            + y_tail_rod     * masses["tail_rod"]
            + y_ht_spar      * m_ht_spar
            + y_ht_elev      * m_ht_rud
            + y_vt_spar      * m_vt_spar
            + y_vt_rud       * m_vt_rud
            + y_pvc          * masses["pvc_tubes"]
            + y_servo_avg    * m_servos
            + y_wing         * masses.get("glass_sheet_wing",   0.0)
            + y_tail_h       * masses.get("glass_sheet_tail_h", 0.0)
            + y_tail_v       * masses.get("glass_sheet_tail_v", 0.0)
            + sensor_mass    * y_sensor
            + w_masses['wing_power'] * y_wiring_wing
            + w_masses['tail_power'] * y_wiring_tail
            + w_masses['signal']     * y_wiring_signal
        ) / m_total
    else:
        y_overall = 0.0

    return {
        'fuselage':           y_fus,
        'battery':            y_batt,
        'battery_bottom':     batt_y0,
        'motors':             y_motor,
        'motor_back':         y_motor_back,
        'wing':               y_wing,
        'wing_ac':            y_ac,       # camber-line height at x/c=0.25; thrust moment reference
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
        'wiring_wing':        y_wiring_wing,
        'wiring_tail':        y_wiring_tail,
        'wiring_signal':      y_wiring_signal,
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
    """Compute total aircraft mass as sum of components.

    Note: 'total' includes a 1.21 integration margin (21%) applied to the
    raw component sum to account for fasteners, adhesives, cable management,
    and miscellaneous hardware not explicitly modelled. Individual component
    entries in the returned dict do NOT include this factor — only 'total'
    does. Do not sum individual entries and expect them to match 'total'.
    """
    if wiring_inputs is None:
        wiring_inputs = WiringInputs()

    m_battery     = propulsion.battery_mass
    m_motors      = propulsion.total_motor_mass
    m_props       = propulsion.inputs.n_props * propulsion.inputs.prop_mass
    m_wing        = wing_mass(sizing, airfoil_path, wing_thickness, materials=materials)
    m_tail_h      = hor_tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_tail_v      = ver_tail_mass(sizing, tail_airfoil_path, tail_thickness, materials=materials)
    m_spar_ht     = structure.mass_spar_ht
    m_control_ht  = structure.mass_control_ht
    m_spar_vt     = structure.mass_spar_vt
    m_control_vt  = structure.mass_control_vt
    m_rod_spar    = structure.mass_spar
    m_rod_aileron = structure.mass_aileron
    m_tail_rod    = structure.mass_t
    m_fuselage    = materials.fuselage.mass(fus.volume_shell)
    x_rod_spar    = _max_tc_x(airfoil_path) * sizing.c_root
    tube_x0       = x_rod_spar - fus.inputs.tube_tail_overlap
    tube_x1       = fus.x_nose + fus.l_nose + fus.box_length - fus.inputs.casing_thickness
    m_pvc         = _pvc_mass(structure, tube_x1 - tube_x0)
    m_servos      = 8 * SERVO_MASS

    x_batt   = 0.25 * sizing.c_root
    x_motor  = 0.0
    x_tail   = 0.25 * sizing.c + sizing.lh
    w_masses = _compute_wiring_masses(sizing, wiring_inputs, x_batt, x_motor, x_tail)

    wing_sheet_area     = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    m_sheet_wing        = materials.sheet.mass(wing_sheet_area)
    m_sheet_tail_h      = materials.sheet.mass(hor_tail_sheet_area)
    m_sheet_tail_v      = materials.sheet.mass(ver_tail_sheet_area)
    m_glass_sheet       = m_sheet_wing + m_sheet_tail_h + m_sheet_tail_v

    # 1.21 = integration margin: fasteners, adhesive, cable management, misc hardware.
    # Applied only to the grand total — individual entries are raw component masses.
    _INTEGRATION_MARGIN = 1.0

    m_total = (
        m_battery + m_motors + m_props + m_wing + m_tail_h + m_tail_v
        + m_rod_spar + m_rod_aileron + m_spar_ht + m_control_ht
        + m_spar_vt + m_control_vt + m_tail_rod + m_fuselage + m_pvc
        + m_servos + m_glass_sheet + sensor_mass + w_masses['total']
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
        'wiring':             w_masses['total'],
        'glass_sheet_wing':   m_sheet_wing,
        'glass_sheet_tail_h': m_sheet_tail_h,
        'glass_sheet_tail_v': m_sheet_tail_v,
        'glass_sheet':        m_glass_sheet,
        'total':              _INTEGRATION_MARGIN * m_total,
    }