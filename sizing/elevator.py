# Positive elevator deflection (elevator down) produces nose-down pitch so CM_deltaE is -ve.
"""Elevator sizing: minimum elevator geometry to trim at cruise.

Procedure
---------
1.  Solve for the tail incidence angle ih from moment equilibrium about the
    CG at delta_e = 0.  Moments are contributed by:
      - Wing-body pitching moment (Cm0, Cmalpha)
      - Thrust: 2 front propellers at y_motor_front height,
                1 back propeller  at y_motor_back  height
      - Tail lift at alpha_h = alpha - epsilon + ih
2.  Fix cE/ch (designer input), derive tau_e from the empirical polynomial,
    then solve for bE/bh from the disturbance requirement:
        bE/bh = Cm_dist / (CLalphah * eta_h * Vh * tau_e * delta_e_max)
3.  Derive SE/Sh = (cE/ch) * (bE/bh)  (rectangular-panel assumption).
4.  Compute all control derivatives and return.

Note on thrust moments
----------------------
Thrust pitching moments (Cm_thrust_front, Cm_thrust_back) are now also
included in the upstream scissor-plot trim balance via `compute_cm_thrust`
in pipeline/helpers.py.  The scissor loop therefore sizes Sh to account for
thrust, so the elevator here is only responsible for disturbance and payload
authority rather than compensating for an under-sized tail.
"""
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import LLTResult
from pipeline.helpers import ScissorData, compute_cm_thrust
from propulsion.sizing import PropulsionResult
from sizing.wing import SizingResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ElevatorInputs:
    trim_mode:                str = "tail_incidence"  # tail_incidence | payload_attachment
    cE_ch:                   float = 0.40   # elevator chord / tail chord [-] (designer choice)
    eta_h:                   float = 0.85   # dynamic-pressure ratio at the tail (Vh/V)²
    Cm_dist:                 float = 0.10   # pitch-moment disturbance the elevator must
                                            # counteract at delta_e_max [-]
    max_deflection_up_deg:   float = 25.0   # elevator-up   deflection limit [deg]
    max_deflection_down_deg: float = 20.0   # elevator-down deflection limit [deg]
    payload_cable_min_angle_deg: float = 0.0       # min cable sweep angle from vertical [deg]
    payload_cable_max_angle_deg: float | None = None  # max angle; None -> 2 * equilibrium angle
    enforce_tail_stall:      bool = True    # raise if max |CLh| exceeds tail Cl_max


@dataclass
class ElevatorGeometry:
    """Derived geometry — output of sizing, never a designer input."""
    bE_bh: float   # elevator span  / tail span  [-]
    cE_ch: float   # elevator chord / tail chord [-]
    SE_Sh: float   # elevator area  / tail area  [-]


@dataclass
class PayloadCableResult:
    """Shared-payload cable equilibrium and pitch-moment envelope."""
    payload_mass_per_drone: float
    payload_weight_per_drone: float
    payload_drag_per_drone: float
    cable_angle_rad: float
    horizontal_tension: float
    vertical_load: float
    attachment_y: float
    vertical_drop: float
    attachment_dx: float
    attachment_angle_rad: float
    Cm_at_equilibrium: float
    sweep_min_angle_rad: float
    sweep_max_angle_rad: float
    worst_pitch_up_angle_rad: float
    worst_pitch_down_angle_rad: float
    Cm_pitch_up: float
    Cm_pitch_down: float


@dataclass
class ElevatorResult:
    inputs:         ElevatorInputs
    geometry:       ElevatorGeometry
    ih:             float           # tail incidence angle from trim [rad]
    Vh:             float           # tail volume coefficient [-]
    Sh_S:           float           # horizontal-tail area ratio [-]
    alpha_h:        float           # tail angle of attack at cruise [rad]
    tau_e:          float           # elevator effectiveness [-]
    CM_deltaE:      float           # dCm/d(delta_e) [1/rad]
    CL_deltaE:      float           # dCL/d(delta_e) [1/rad]
    CLh_deltaE:     float           # dCLh/d(delta_e) [1/rad]
    CLh_cruise:     float           # tail section CL at cruise [–]
    CLh_at_max_up:  float           # tail section CL at maximum elevator-up deflection [–]
    CLh_at_max_down: float          # tail section CL at maximum elevator-down deflection [–]
    Cl_max:         float           # tail airfoil maximum section Cl [–]
    Cm_wing_body:   float
    Cm_tail_base:   float
    Cm_tail_incidence: float
    Cm_tail_total:  float
    Cm_thrust_front: float
    Cm_thrust_back:  float
    Cm_thrust_total: float
    driving_constraint: str
    bE_bh_required: float
    bE_bh_stall_limit: float
    Cm_payload:     float
    payload_cable:  PayloadCableResult
    y_cg:           dict[str, float]
    alpha:          float   # wing cruise AoA [rad]
    epsilon:        float   # downwash at tail [rad]


# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationships
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """Control-surface effectiveness τ from chord ratio cf/c (empirical fit)."""
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


def compute_payload_cable(
    sizing: SizingResult,
    y_cg: dict[str, float],
    inputs: ElevatorInputs,
    *,
    Cm_trim: float = 0.0,
) -> PayloadCableResult:
    """Compute the per-drone payload cable angle and pitch moment envelope.

    The attachment is placed so the nominal equilibrium cable line passes
    through the CG. Therefore the pitch moment is zero at the equilibrium
    angle. The envelope sweeps cable angles from vertical to the same angular
    excursion on the other side of equilibrium.
    """
    s = sizing
    n_drones = max(float(s.inputs.n_drones), 1.0)
    payload_mass = s.inputs.m_payload / n_drones
    payload_weight = payload_mass * 9.80665
    payload_drag = s.q_cruise * s.inputs.Cd_payload * s.inputs.S_payload / n_drones
    cable_angle = float(np.arctan2(payload_drag, payload_weight))

    attachment_y = y_cg["pvc_tubes"]
    vertical_drop = max(y_cg["overall"] - attachment_y, 0.0)
    denom = s.q_cruise * s.Sw * s.c
    if vertical_drop > 0.0 and payload_weight > 0.0 and denom > 0.0:
        tan_attachment = np.tan(cable_angle) - (
            Cm_trim * denom / (payload_weight * vertical_drop)
        )
    elif abs(Cm_trim) > 1.0e-12:
        raise ValueError(
            "Payload attachment trim is impossible because the cable attachment "
            "has no vertical arm below the CG."
        )
    else:
        tan_attachment = np.tan(cable_angle)
    attachment_angle = float(np.arctan(tan_attachment))
    attachment_dx = vertical_drop * tan_attachment

    sweep_min_angle = np.radians(inputs.payload_cable_min_angle_deg)
    if inputs.payload_cable_max_angle_deg is None:
        sweep_max_angle = 2.0 * cable_angle
    else:
        sweep_max_angle = np.radians(inputs.payload_cable_max_angle_deg)
    sweep_min_angle = float(np.clip(sweep_min_angle, 0.0, np.radians(89.0)))
    sweep_max_angle = float(np.clip(sweep_max_angle, 0.0, np.radians(89.0)))
    if sweep_max_angle < sweep_min_angle:
        raise ValueError(
            "payload_cable_max_angle_deg must be greater than or equal to "
            "payload_cable_min_angle_deg."
        )
    cable_angles = np.linspace(sweep_min_angle, sweep_max_angle, 401)

    if denom > 0.0:
        Cm_absolute = (
            payload_weight
            * vertical_drop
            * (np.tan(cable_angles) - tan_attachment)
            / denom
        )
    else:
        Cm_absolute = np.zeros_like(cable_angles)
    Cm_at_equilibrium = (
        payload_weight
        * vertical_drop
        * (np.tan(cable_angle) - tan_attachment)
        / denom
        if denom > 0.0 else 0.0
    )
    Cm_sweep = Cm_absolute - Cm_at_equilibrium

    i_up = int(np.argmax(Cm_sweep))
    i_down = int(np.argmin(Cm_sweep))

    return PayloadCableResult(
        payload_mass_per_drone=payload_mass,
        payload_weight_per_drone=payload_weight,
        payload_drag_per_drone=payload_drag,
        cable_angle_rad=cable_angle,
        horizontal_tension=payload_drag,
        vertical_load=payload_weight,
        attachment_y=attachment_y,
        vertical_drop=vertical_drop,
        attachment_dx=attachment_dx,
        attachment_angle_rad=attachment_angle,
        Cm_at_equilibrium=Cm_at_equilibrium,
        sweep_min_angle_rad=sweep_min_angle,
        sweep_max_angle_rad=sweep_max_angle,
        worst_pitch_up_angle_rad=float(cable_angles[i_up]),
        worst_pitch_down_angle_rad=float(cable_angles[i_down]),
        Cm_pitch_up=float(Cm_sweep[i_up]),
        Cm_pitch_down=float(Cm_sweep[i_down]),
    )


def _span_limit_for_tail_stall(
    CLh_cruise: float,
    CLh_delta_per_span: float,
    delta_e: float,
    Cl_max: float,
) -> float:
    """Largest bE/bh allowed by |CLh_cruise + CLh_delta*delta_e*bE_bh| <= Cl_max."""
    k = CLh_delta_per_span * delta_e
    if abs(k) < 1.0e-12:
        return float("inf")

    lo = (-Cl_max - CLh_cruise) / k
    hi = ( Cl_max - CLh_cruise) / k
    if lo > hi:
        lo, hi = hi, lo
    if hi < 0.0:
        return float("-inf")
    return hi


# ---------------------------------------------------------------------------
# Main sizing routine
# ---------------------------------------------------------------------------

def run(
    sizing:       SizingResult,
    scissor:      ScissorData,
    llt:          LLTResult,
    propulsion:   PropulsionResult,
    wing_polar:   AirfoilPolar,
    tail_polar:   AirfoilPolar,
    y_cg:         dict[str, float],
    inputs:       ElevatorInputs | None = None,
) -> ElevatorResult:
    """
    Size the elevator by:
      1. Solving for ih from moment equilibrium at delta_e = 0.
      2. Fixing cE/ch (from inputs), deriving tau_e, then solving for bE/bh
         from the disturbance requirement.

    Parameters
    ----------
    sizing      : converged wing/tail sizing result
    scissor     : scissor-plot stability data (Sh already accounts for thrust
                  moments via compute_scissor_data)
    llt         : LLT result carrying the cruise angle of attack
    propulsion  : propulsion result for thrust pitching moment
    wing_polar  : wing airfoil polar for zero-lift angle
    tail_polar  : tail airfoil polar for stall coefficient checks
    y_cg        : vertical CG dict from compute_y_cg (keys: 'overall',
                  'wing_ac', 'motors', 'motor_back', 'pvc_tube_bottom')
    inputs      : ElevatorInputs overrides; defaults used when None

    Returns
    -------
    ElevatorResult containing ih, converged elevator geometry, and
    associated control derivatives.
    """
    if inputs is None:
        inputs = ElevatorInputs()
    i = inputs
    s = sizing
    trim_mode = i.trim_mode.lower()
    if trim_mode not in {"tail_incidence", "payload_attachment"}:
        raise ValueError(
            "elevator.trim_mode must be 'tail_incidence' or "
            "'payload_attachment'."
        )
    # ------------------------------------------------------------------
    # Aero state from the converged design
    # ------------------------------------------------------------------
    CLalpha  = scissor.CL_alpha_w
    CLalphah = scissor.CL_alpha_h
    Cmalpha  = -scissor.CL_alpha_A_h * scissor.SM
    Cm0      = scissor.Cm_ac
    AR       = s.inputs.AR
    alpha    = llt.alpha_root

    Vh   = s.Sh * s.lh / (s.Sw * s.c)
    Sh_S = s.Sh / s.Sw

    # ------------------------------------------------------------------
    # Downwash at cruise
    # ------------------------------------------------------------------
    CL0      = -CLalpha * wing_polar.alpha_L0
    epsilon = scissor.dep_da * alpha  # epsilon0 ≈ 0 for symmetric tail (alpha_L0_h = 0)

    # ------------------------------------------------------------------
    # Thrust pitching moments about the wing AC
    # 2 front propellers at y_cg["motors"], 1 back at y_cg["motor_back"].
    # Z_T positive when motor is above the wing AC.
    # ------------------------------------------------------------------
    y_thrust_ref = y_cg["wing_ac"]
    Z_T_front = y_cg["motors"]     - y_thrust_ref
    Z_T_back  = y_cg["motor_back"] - y_thrust_ref

    T_per_prop = propulsion.thrust_cruise_per_prop
    q          = s.q_cruise

    Cm_thrust_front, Cm_thrust_back = compute_cm_thrust(
        T_per_prop, Z_T_front, Z_T_back, q, s.Sw, s.c,
    )
    Cm_thrust = Cm_thrust_front + Cm_thrust_back

    # ------------------------------------------------------------------
    # Solve for ih from moment equilibrium at delta_e = 0
    #
    # Cm_total = Cm0 + Cmalpha*(alpha - alpha_L0)          [wing-body]
    #          + Cm_thrust                                  [propellers]
    #          - CLalphah * eta_h * Vh                      [aft tail]
    #            * (alpha - epsilon + ih - alpha_L0_h)
    #          = 0
    #
    # Positive tail lift acts aft of the CG and therefore creates a
    # nose-down pitching moment.
    #
    # alpha_L0_h = 0 for a symmetric tail airfoil.
    # ------------------------------------------------------------------
    alpha_L0   = wing_polar.alpha_L0
    alpha_L0_h = 0.0   # symmetric tail airfoil assumption

    Cm_wing_body = Cm0 + Cmalpha * (alpha - alpha_L0)
    Cm_tail_per_alpha = -CLalphah * i.eta_h * Vh
    Cm_tail_base = Cm_tail_per_alpha * (alpha - epsilon - alpha_L0_h)

    if trim_mode == "payload_attachment":
        ih = -(alpha - epsilon - alpha_L0_h)
    else:
        ih = -(Cm_wing_body + Cm_thrust + Cm_tail_base) / Cm_tail_per_alpha
    Cm_tail_incidence = Cm_tail_per_alpha * ih
    Cm_tail_total = Cm_tail_base + Cm_tail_incidence
    Cm_payload_trim = (
        -(Cm_wing_body + Cm_thrust + Cm_tail_total)
        if trim_mode == "payload_attachment" else 0.0
    )

    # Tail angle of attack at cruise with solved ih
    alpha_h = alpha - epsilon + ih - alpha_L0_h

    # ------------------------------------------------------------------
    # Payload disturbance (from config.yaml sizing.m_payload)
    # ------------------------------------------------------------------
    payload_cable = compute_payload_cable(sizing, y_cg, i, Cm_trim=Cm_payload_trim)
    Cm_payload = max(
        abs(payload_cable.Cm_pitch_up),
        abs(payload_cable.Cm_pitch_down),
    )

    # ------------------------------------------------------------------
    # Elevator sizing
    #
    # 1. Fix cE/ch (designer input) → tau_e from empirical polynomial.
    # 2. Solve for bE/bh from disturbance requirement using the magnitude of
    #    the maximum elevator-up deflection.
    #      Cm_dist = CLalphah * eta_h * Vh * tau_e * |delta_e_up| * bE_bh
    #      → bE_bh = Cm_dist / (CLalphah * eta_h * Vh * tau_e * |delta_e_up|)
    # ------------------------------------------------------------------
    delta_e_max = np.radians(i.max_deflection_up_deg)
    delta_e_up = -delta_e_max
    delta_e_down = np.radians(i.max_deflection_down_deg)

    tau_e = tau_from_chord_ratio(i.cE_ch)

    control_power = CLalphah * i.eta_h * Vh * tau_e
    bE_bh_dist = i.Cm_dist / (control_power * delta_e_max)
    bE_bh_payload_up = max(payload_cable.Cm_pitch_up, 0.0) / (
        control_power * delta_e_down
    )
    bE_bh_payload_down = max(-payload_cable.Cm_pitch_down, 0.0) / (
        control_power * delta_e_max
    )
    bE_bh_payload = max(bE_bh_payload_up, bE_bh_payload_down)
    driving_constraint = "payload" if bE_bh_payload > bE_bh_dist else "disturbance"
    bE_bh_required = max(
        bE_bh_dist,
        bE_bh_payload,
    )

    if bE_bh_required > 1.0:
        raise ValueError(
            f"Elevator sizing failed: required bE/bh = {bE_bh_required:.3f} > 1.0.\n"
            "The full tail span is insufficient to counteract Cm_dist.\n"
            "Consider increasing cE/ch, increasing tail volume, or reducing Cm_dist."
        )

    CLh_cruise = CLalphah * alpha_h
    Cl_max = tail_polar.Cl_max
    CLh_delta_per_span = CLalphah * tau_e
    bE_bh_stall_limit = min(
        1.0,
        _span_limit_for_tail_stall(
            CLh_cruise, CLh_delta_per_span, delta_e_up, Cl_max,
        ),
        _span_limit_for_tail_stall(
            CLh_cruise, CLh_delta_per_span, delta_e_down, Cl_max,
        ),
    )

    if i.enforce_tail_stall:
        if bE_bh_stall_limit < 0.0:
            raise ValueError(
                "Tail sizing failed: the trimmed tail is already beyond the "
                "tail airfoil |Cl_max| before elevator span is applied.\n"
                f"  CLh_cruise = {CLh_cruise:.3f}\n"
                f"  |Cl_max| = {Cl_max:.3f}"
            )
        if bE_bh_required > bE_bh_stall_limit + 1.0e-9:
            raise ValueError(
                "Elevator sizing failed: required span exceeds the largest "
                "span that avoids tail stall at max deflection.\n"
                f"  required bE/bh = {bE_bh_required:.3f}\n"
                f"  stall-limited bE/bh = {bE_bh_stall_limit:.3f}\n"
                f"  CLh_cruise = {CLh_cruise:.3f}, |Cl_max| = {Cl_max:.3f}"
            )
        bE_bh = bE_bh_stall_limit
    else:
        bE_bh = bE_bh_required

    # ------------------------------------------------------------------
    # Tail stall check: ensure cruise incidence plus maximum elevator deflection
    # does not demand a tail lift coefficient above the airfoil's Cl_max.
    # Elevator influence is scaled by the elevator span fraction bE/bh.
    # ------------------------------------------------------------------
    CLh_at_max_up = CLh_cruise + CLalphah * tau_e * delta_e_up * bE_bh
    CLh_at_max_down = CLh_cruise + CLalphah * tau_e * delta_e_down * bE_bh
    CLh_max_required = max(abs(CLh_at_max_up), abs(CLh_at_max_down))

    if i.enforce_tail_stall and CLh_max_required > Cl_max + 1.0e-9:
        raise ValueError(
            "Tail sizing failed: required tail lift coefficient at one of the "
            "control extremes exceeds the tail airfoil |Cl_max|.\n"
            f"  CLh({i.max_deflection_up_deg:.1f}° elevator up)   = {CLh_at_max_up:.3f}\n"
            f"  CLh({i.max_deflection_down_deg:.1f}° elevator down) = {CLh_at_max_down:.3f}\n"
            f"  |Cl_max| = {Cl_max:.3f}\n"
            "Reduce trim/elevator demand, choose a higher-Cl tail airfoil, "
            "or increase tail volume."
        )

    # ------------------------------------------------------------------
    # Final geometry and derivatives
    # ------------------------------------------------------------------
    geometry = ElevatorGeometry(
        bE_bh = bE_bh,
        cE_ch = i.cE_ch,
        SE_Sh = i.cE_ch * bE_bh,
    )

    CM_deltaE  = -CLalphah * i.eta_h * Vh * bE_bh * tau_e
    CL_deltaE  =  CLalphah * i.eta_h * Sh_S * bE_bh * tau_e
    CLh_deltaE =  CLalphah * tau_e

    return ElevatorResult(
        inputs          = inputs,
        geometry        = geometry,
        ih              = ih,
        Vh              = Vh,
        Sh_S            = Sh_S,
        alpha_h         = alpha_h,
        tau_e           = tau_e,
        CM_deltaE       = CM_deltaE,
        CL_deltaE       = CL_deltaE,
        CLh_deltaE      = CLh_deltaE,
        CLh_cruise      = CLh_cruise,
        CLh_at_max_up   = CLh_at_max_up,
        CLh_at_max_down = CLh_at_max_down,
        Cl_max          = Cl_max,
        Cm_wing_body    = Cm_wing_body,
        Cm_tail_base    = Cm_tail_base,
        Cm_tail_incidence = Cm_tail_incidence,
        Cm_tail_total   = Cm_tail_total,
        Cm_thrust_front = Cm_thrust_front,
        Cm_thrust_back  = Cm_thrust_back,
        Cm_thrust_total = Cm_thrust,
        Cm_payload      = Cm_payload,
        payload_cable   = payload_cable,
        driving_constraint = driving_constraint,
        bE_bh_required  = bE_bh_required,
        bE_bh_stall_limit = bE_bh_stall_limit,
        alpha           = alpha,
        epsilon         = epsilon,
        y_cg            = y_cg,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summary(r: ElevatorResult) -> None:
    i = r.inputs
    g = r.geometry
    print("\n================ CG (vertical axis) =================")
    print("Sign convention: +y upward")
    print("====================================================")

    print(f"  Front motors y-position : {r.y_cg['motors']:.4f}")
    print(f"  Back motor y-position   : {r.y_cg['motor_back']:.4f}")
    print(f"  Wing AC y-position      : {r.y_cg['wing_ac']:.4f}")
    print(f"  Aircraft CG (overall)    : {r.y_cg['overall']:.4f}")
    print("====================================================\n")

    print("\n================ Pitch Moment Sign Convention ================")
    print("  Cm > 0 : nose-up pitch")
    print("  Cm < 0 : nose-down pitch")
    print("  Elevator TE-down (positive deflection) → nose-down moment")
    print("==============================================================\n")

    print("----- Pitching Moment Breakdown (about wing AC) -----")
    print(f"  Trim mode                         : {r.inputs.trim_mode}")
    print(f"  Wing-body moment (Cm0 + α term) : {r.Cm_wing_body:+.5f}")
    print(f"  Tail moment (α - ε - αL0,h)     : {r.Cm_tail_base:+.5f}")
    print(f"  Tail moment from incidence ih   : {r.Cm_tail_incidence:+.5f}")
    print(f"  Tail moment total               : {r.Cm_tail_total:+.5f}")
    print(f"  Thrust (front 2 props)          : {r.Cm_thrust_front:+.5f}")
    print(f"  Thrust (rear prop)              : {r.Cm_thrust_back:+.5f}")
    print(f"  Payload trim moment             : {r.payload_cable.Cm_at_equilibrium:+.5f}")
    print(f"  ----------------------------------------------")
    print(f"  Cruise moment buildup total     : "
        f"{(r.Cm_wing_body + r.Cm_tail_total + r.Cm_thrust_total + r.payload_cable.Cm_at_equilibrium):+.5f}")

    print("\n----- Tail Contribution at cruise (trim) -----")
    print(f"  Tail AoA α_h             : {np.degrees(r.alpha_h):+.3f} deg")
    print(f"  Tail CL contribution     : {r.CLh_cruise:+.5f}")
    print(f"  Tail Cm contribution     : {r.Cm_tail_total:+.5f}\n")
    print("\n----- Tail AoA Breakdown -----")
    print(f"  {'Component':<30} {'Value [deg]':>12}")
    print(f"  {'-'*43}")
    print(f"  {'Wing AoA (α)':<30} {np.degrees(r.alpha):>+12.3f}")
    print(f"  {'Downwash (-ε)':<30} {np.degrees(-r.epsilon):>+12.3f}")
    print(f"  {'Tail incidence (ih)':<30} {np.degrees(r.ih):>+12.3f}")
    print(f"  {'-'*43}")
    print(f"  {'Tail AoA (α_h)':<30} {np.degrees(r.alpha_h):>+12.3f}")
    print("----- Payload Contribution -----")
    pc = r.payload_cable
    print(f"  Payload mass / drone     : {pc.payload_mass_per_drone:.3f} kg")
    print(f"  Payload drag / drone     : {pc.payload_drag_per_drone:.3f} N")
    print(f"  Payload weight / drone   : {pc.payload_weight_per_drone:.3f} N")
    print(f"  Cable angle from vertical: {np.degrees(pc.cable_angle_rad):+.3f} deg")
    print(f"  Tube centerline y        : {pc.attachment_y:.4f} m")
    print(f"  CG-to-attach vertical    : {pc.vertical_drop:.4f} m")
    print(f"  CG-to-attach x offset    : {pc.attachment_dx:.4f} m")
    print(f"  Attachment angle from vertical: {np.degrees(pc.attachment_angle_rad):+.3f} deg")
    print(f"  Cm at equilibrium angle  : {pc.Cm_at_equilibrium:+.5f}")
    print(f"  Cable sweep              : "
          f"{np.degrees(pc.sweep_min_angle_rad):.3f} to "
          f"{np.degrees(pc.sweep_max_angle_rad):.3f} deg")
    print(f"  Worst pitch-up angle     : "
          f"{np.degrees(pc.worst_pitch_up_angle_rad):+.3f} deg")
    print(f"  Worst pitch-up Cm        : {pc.Cm_pitch_up:+.5f}")
    print(f"  Worst pitch-down angle   : "
          f"{np.degrees(pc.worst_pitch_down_angle_rad):+.3f} deg")
    print(f"  Worst pitch-down Cm      : {pc.Cm_pitch_down:+.5f}")
    print(f"  Cm_payload envelope      : {r.Cm_payload:+.5f}")

    print("\n----- Elevator Sizing Driver -----")
    print(f"  Active constraint         : {r.driving_constraint}")
    print(f"  Required bE/bh            : {r.bE_bh_required:.4f}")
    print(f"  Stall-limited max bE/bh   : {r.bE_bh_stall_limit:.4f}")
    print("----- Elevator Sensitivity -----")
    print("  Sensitivity inputs:")
    CL_alpha_h = r.CLh_deltaE / r.tau_e
    print(f"    CL_alpha_h              : {CL_alpha_h:.4f}  1/rad")
    print(f"    CL_alpha_h unit check   : {CL_alpha_h * np.pi / 180.0:.5f}  1/deg")
    print(f"    eta_h                   : {r.inputs.eta_h:.4f}")
    print(f"    Vh = Sh*lh/(Sw*c)       : {r.Vh:.4f}")
    print(f"    Sh/S                    : {r.Sh_S:.4f}")
    print(f"    bE/bh                   : {g.bE_bh:.4f}")
    print(f"    tau_e                   : {r.tau_e:.4f}")
    print(f"    delta_e up/down         : {-r.inputs.max_deflection_up_deg:.2f}, "
          f"{r.inputs.max_deflection_down_deg:.2f} deg")
    print(f"  CM_δe                    : {r.CM_deltaE:+.5f}  1/rad")
    print(f"  CL_δe                    : {r.CL_deltaE:+.5f}  1/rad")
    print(f"  CLh_δe                   : {r.CLh_deltaE:+.5f}  1/rad")

    print("  Formulas:")
    print("    CM_delta_e  = -CL_alpha_h*eta_h*Vh*bE/bh*tau_e")
    print("    CL_delta_e  =  CL_alpha_h*eta_h*Sh/S*bE/bh*tau_e")
    print("    CLh_delta_e =  CL_alpha_h*tau_e")

    print("\n----- Tail Extremes Check -----")
    print(f"  CLh (elevator up max)    : {r.CLh_at_max_up:+.5f}")
    print(f"  CLh (elevator down max)  : {r.CLh_at_max_down:+.5f}")
    print(f"  Airfoil Cl_max           : {r.Cl_max:+.5f}")

    print("\n----- Elevator Geometry -----")
    print(f"  cE/ch                    : {g.cE_ch:.4f}")
    print(f"  bE/bh                    : {g.bE_bh:.4f}")
    print(f"  SE/Sh                    : {g.SE_Sh:.4f}")
    print(f"  tau_e                    : {r.tau_e:.4f}")
