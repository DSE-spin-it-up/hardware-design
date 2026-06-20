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

Note on rear-engine-out sizing
-------------------------------
The rear propeller is mounted on top of the vertical tail. If it fails, its
pitching-moment contribution (Cm_thrust_back) vanishes from the trim
balance, and the elevator must be able to retrim against this lost moment.
This is treated as an independent sizing case alongside disturbance,
payload, and gust (see `bE_bh_engine_out` below).
"""
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import FlightCondition, LLTResult, WingGeometry, solve_llt
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
    tail_clmax_fraction:     float = 0.90   # allowed fraction of finite-tail CL_max [-]
    enforce_tail_stall:      bool = True    # raise if cruise |CLh| exceeds tail Cl_max


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
    q_sizing: float
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
    tail_CL_max_3d:  float          # finite-tail maximum CL from LLT [-]
    tail_CL_limit:   float          # allowed finite-tail CL limit after fraction [-]
    Cl_max:         float           # compatibility alias for tail_CL_limit [-]
    Cm_wing_body:   float
    Cm_tail_base:   float
    Cm_tail_incidence: float
    Cm_tail_total:  float
    Cm_thrust_front: float
    Cm_thrust_back:  float
    Cm_thrust_total: float
    driving_constraint: str
    bE_bh_required: float
    bE_bh_disturbance: float
    bE_bh_payload: float
    bE_bh_gust: float
    bE_bh_engine_out: float
    bE_bh_stall_limit: float
    Cm_payload:     float
    Cm_gust:        float
    Cm_engine_out:  float
    gust_speed:     float
    gust_case_speed: float
    payload_cable:  PayloadCableResult
    y_cg:           dict[str, float]
    alpha:          float   # wing cruise AoA [rad]
    epsilon:        float   # downwash at tail [rad]


# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationships
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """Aileron effectiveness τ from chord ratio cf/c (empirical polynomial fit)."""
    if not (0.0 < cf_c < 0.7):
        raise ValueError(
            f"cf/c = {cf_c:.3f} out of valid range (0, 0.7)."
        )
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))

def compute_payload_cable(
    sizing: SizingResult,
    y_cg: dict[str, float],
    inputs: ElevatorInputs,
    *,
    Cm_trim: float = 0.0,
    q_sizing: float | None = None,
    payload_max_tension: float | None = None,
) -> PayloadCableResult:
    s = sizing
    n_drones = max(float(s.inputs.n_drones), 1.0)
    q_size = s.q_cruise if q_sizing is None else q_sizing
    payload_mass = s.inputs.m_payload / n_drones
    payload_weight = payload_mass * 9.80665
    payload_drag = q_size * s.inputs.Cd_payload * s.inputs.S_payload / n_drones
    cable_angle = float(np.arctan2(payload_drag, payload_weight))

    if payload_max_tension is not None and payload_max_tension < 0.0:
        raise ValueError("payload_max_tension must be non-negative.")

    attachment_y = y_cg["pvc_tubes"]
    vertical_drop = max(y_cg["overall"] - attachment_y, 0.0)
    denom = q_size * s.Sw * s.c

    # Equilibrium horizontal tension (used for trim and for Cm_at_equilibrium)
    horizontal_tension_equilibrium = payload_drag
    horizontal_trim = horizontal_tension_equilibrium
    if vertical_drop > 0.0 and denom > 0.0:
        horizontal_trim = horizontal_tension_equilibrium - Cm_trim * denom / vertical_drop
        tan_attachment = (
            horizontal_trim / payload_weight
            if payload_weight > 0.0 else np.tan(cable_angle)
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

    Cm_at_equilibrium = (
        vertical_drop * (horizontal_tension_equilibrium - horizontal_trim) / denom
        if denom > 0.0 else 0.0
    )

    # Worst-case moment: max tension pulls horizontally over the full vertical arm.
    # Both pitch-up and pitch-down are symmetric about equilibrium.
    if payload_max_tension is not None and denom > 0.0:
        Cm_worst = payload_max_tension * vertical_drop / denom
    elif denom > 0.0:
        # Fallback: use payload weight as a conservative proxy
        Cm_worst = payload_weight * vertical_drop / denom
    else:
        Cm_worst = 0.0

    Cm_pitch_up   =  Cm_worst
    Cm_pitch_down = -Cm_worst

    return PayloadCableResult(
        payload_mass_per_drone=payload_mass,
        q_sizing=q_size,
        payload_weight_per_drone=payload_weight,
        payload_drag_per_drone=payload_drag,
        cable_angle_rad=cable_angle,
        horizontal_tension=payload_max_tension if payload_max_tension is not None else payload_drag,
        vertical_load=payload_weight,
        attachment_y=attachment_y,
        vertical_drop=vertical_drop,
        attachment_dx=attachment_dx,
        attachment_angle_rad=attachment_angle,
        Cm_at_equilibrium=Cm_at_equilibrium,
        sweep_min_angle_rad=0.0,   # no longer meaningful, kept for dataclass compat
        sweep_max_angle_rad=0.0,
        worst_pitch_up_angle_rad=float(np.pi / 2),   # implicit: cable horizontal
        worst_pitch_down_angle_rad=float(np.pi / 2),
        Cm_pitch_up=Cm_pitch_up,
        Cm_pitch_down=Cm_pitch_down,
    )


def _finite_tail_cl_limit_llt(sizing: SizingResult, tail_polar: AirfoilPolar) -> float:
    """Finite-tail CL limit from LLT at the first local section Cl limit."""
    geom = WingGeometry(
        b=sizing.bh,
        S=sizing.Sh,
        taper=sizing.inputs.lam_t,
    )
    llt0 = solve_llt(
        geom,
        tail_polar,
        FlightCondition(V_inf=sizing.inputs.V_cruise, rho=sizing.rho, alpha_root=0.0),
    )
    llt1 = solve_llt(
        geom,
        tail_polar,
        FlightCondition(V_inf=sizing.inputs.V_cruise, rho=sizing.rho, alpha_root=1.0),
    )

    cl0 = llt0.Cl_local
    cl_slope = llt1.Cl_local - cl0
    cl_section_max = float(np.max(tail_polar.Cl))
    cl_section_min = float(np.min(tail_polar.Cl))

    candidates: list[float] = []
    positive_alpha = np.divide(
        cl_section_max - cl0,
        cl_slope,
        out=np.full_like(cl0, np.inf),
        where=cl_slope > 1.0e-12,
    )
    positive_alpha = positive_alpha[positive_alpha > 0.0]
    if positive_alpha.size:
        llt_pos = solve_llt(
            geom,
            tail_polar,
            FlightCondition(
                V_inf=sizing.inputs.V_cruise,
                rho=sizing.rho,
                alpha_root=float(np.min(positive_alpha)),
            ),
        )
        candidates.append(abs(llt_pos.CL))

    negative_alpha = np.divide(
        cl_section_min - cl0,
        cl_slope,
        out=np.full_like(cl0, -np.inf),
        where=cl_slope > 1.0e-12,
    )
    negative_alpha = negative_alpha[negative_alpha < 0.0]
    if negative_alpha.size:
        llt_neg = solve_llt(
            geom,
            tail_polar,
            FlightCondition(
                V_inf=sizing.inputs.V_cruise,
                rho=sizing.rho,
                alpha_root=float(np.max(negative_alpha)),
            ),
        )
        candidates.append(abs(llt_neg.CL))

    if not candidates:
        raise RuntimeError("Could not derive finite-tail CL limit from LLT.")
    return min(candidates)


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
    *,
    q_sizing:     float | None = None,
    payload_max_tension: float | None = None,
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
    epsilon = scissor.dep_da * (alpha - wing_polar.alpha_L0)

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
    q_size = s.q_cruise if q_sizing is None else q_sizing
    payload_cable = compute_payload_cable(
        sizing,
        y_cg,
        i,
        Cm_trim=Cm_payload_trim,
        q_sizing=q_size,
        payload_max_tension=payload_max_tension,
    )
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
    gust_speed = max(float(s.inputs.gust_speed), 0.0)
    gust_cases: list[tuple[float, float, float]] = []
    for V_case in (
        max(s.inputs.V_cruise - gust_speed, 1.0e-6),
        s.inputs.V_cruise + gust_speed,
    ):
        q_ratio = (V_case / s.inputs.V_cruise) ** 2
        dCm = (q_ratio - 1.0) * (Cm_wing_body + Cm_tail_total)
        Cm_tail_required = -dCm
        if Cm_tail_required >= 0.0:
            bE_bh_case = Cm_tail_required / (
                control_power * q_ratio * delta_e_max
            )
        else:
            bE_bh_case = -Cm_tail_required / (
                control_power * q_ratio * delta_e_down
            )
        gust_cases.append((float(bE_bh_case), float(abs(Cm_tail_required)), float(V_case)))
    bE_bh_gust, Cm_gust, gust_case_speed = max(gust_cases, key=lambda case: case[0])

    # ------------------------------------------------------------------
    # Rear-propeller (mounted on top of the vertical tail) failure case.
    #
    # If the rear motor fails, its pitching-moment contribution
    # (Cm_thrust_back) vanishes from the trim balance. The elevator must
    # be able to retrim against this lost moment — an independent sizing
    # case from disturbance, payload, and gust.
    #
    #   dCm = -Cm_thrust_back        (moment lost when the engine fails)
    #   Cm_tail_required = -dCm = Cm_thrust_back
    # ------------------------------------------------------------------
    Cm_engine_out = float(Cm_thrust_back)
    if Cm_engine_out >= 0.0:
        bE_bh_engine_out = Cm_engine_out / (control_power * delta_e_max)
    else:
        bE_bh_engine_out = -Cm_engine_out / (control_power * delta_e_down)

    constraint_requirements = {
        "disturbance": bE_bh_dist,
        "payload": bE_bh_payload,
        "airspeed_gust": bE_bh_gust,
        "rear_engine_out": bE_bh_engine_out,
    }
    driving_constraint = max(constraint_requirements, key=constraint_requirements.get)
    bE_bh_required = max(
        bE_bh_dist,
        bE_bh_payload,
        bE_bh_gust,
        bE_bh_engine_out,
    )

    if bE_bh_required > 1.0:
        raise ValueError(
            f"Elevator sizing failed: required bE/bh = {bE_bh_required:.3f} > 1.0.\n"
            f"The full tail span is insufficient for the {driving_constraint} constraint.\n"
            "Consider increasing cE/ch, increasing tail volume, or reducing the active demand."
        )

    CLh_cruise = CLalphah * alpha_h
    tail_CL_max_3d = _finite_tail_cl_limit_llt(s, tail_polar)
    tail_CL_limit = i.tail_clmax_fraction * tail_CL_max_3d
    Cl_max = tail_CL_limit
    bE_bh_stall_limit = 1.0

    if i.enforce_tail_stall and abs(CLh_cruise) > tail_CL_limit + 1.0e-9:
        raise ValueError(
            "Tail sizing failed: the undeflected cruise tail lift coefficient "
            "exceeds the allowed finite-tail |CL|.\n"
            f"  CLh_cruise = {CLh_cruise:.3f}\n"
            f"  finite-tail CL_max = {tail_CL_max_3d:.3f}\n"
            f"  allowed fraction = {i.tail_clmax_fraction:.3f}\n"
            f"  allowed |CLh| = {tail_CL_limit:.3f}"
        )
    bE_bh = bE_bh_required

    # ------------------------------------------------------------------
    # Deflected-elevator tail CL is diagnostic only. The finite-tail CL limit
    # comes from the undeflected tail polar, so only CLh_cruise is constrained.
    # ------------------------------------------------------------------
    CLh_at_max_up = CLh_cruise + CLalphah * tau_e * delta_e_up * bE_bh
    CLh_at_max_down = CLh_cruise + CLalphah * tau_e * delta_e_down * bE_bh


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
        tail_CL_max_3d  = tail_CL_max_3d,
        tail_CL_limit   = tail_CL_limit,
        Cl_max          = Cl_max,
        Cm_wing_body    = Cm_wing_body,
        Cm_tail_base    = Cm_tail_base,
        Cm_tail_incidence = Cm_tail_incidence,
        Cm_tail_total   = Cm_tail_total,
        Cm_thrust_front = Cm_thrust_front,
        Cm_thrust_back  = Cm_thrust_back,
        Cm_thrust_total = Cm_thrust,
        Cm_payload      = Cm_payload,
        Cm_gust         = Cm_gust,
        Cm_engine_out   = Cm_engine_out,
        gust_speed      = gust_speed,
        gust_case_speed = gust_case_speed,
        payload_cable   = payload_cable,
        driving_constraint = driving_constraint,
        bE_bh_required  = bE_bh_required,
        bE_bh_disturbance = bE_bh_dist,
        bE_bh_payload   = bE_bh_payload,
        bE_bh_gust      = bE_bh_gust,
        bE_bh_engine_out = bE_bh_engine_out,
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
    print(f"  Payload sizing q         : {pc.q_sizing:.2f} Pa")
    print(f"  Payload drag / drone     : {pc.payload_drag_per_drone:.3f} N")
    print(f"  Payload weight / drone   : {pc.payload_weight_per_drone:.3f} N")
    print(f"  Cable angle from vertical: {np.degrees(pc.cable_angle_rad):+.3f} deg")
    print(f"  Tube centerline y        : {pc.attachment_y:.4f} m")
    print(f"  CG-to-attach vertical    : {pc.vertical_drop:.4f} m")
    print(f"  CG-to-attach x offset    : {pc.attachment_dx:.4f} m")
    print(f"  Attachment angle from vertical: {np.degrees(pc.attachment_angle_rad):+.3f} deg")
    print(f"  Cm at equilibrium angle  : {pc.Cm_at_equilibrium:+.5f}")
    print(f"  Max cable tension        : {pc.horizontal_tension:.3f} N")
    print(f"  Vertical moment arm      : {pc.vertical_drop:.4f} m")
    print(f"  Worst-case |Cm_payload|  : {pc.Cm_pitch_up:+.5f}")
    print(f"  Worst pitch-down Cm      : {pc.Cm_pitch_down:+.5f}")
    print(f"  Cm_payload envelope      : {r.Cm_payload:+.5f}")

    print("\n----- Elevator Sizing Driver -----")
    print(f"  Active constraint         : {r.driving_constraint}")
    print(f"  Required bE/bh            : {r.bE_bh_required:.4f}")
    print(f"    disturbance bE/bh       : {r.bE_bh_disturbance:.4f}")
    print(f"    payload bE/bh           : {r.bE_bh_payload:.4f}")
    print(f"    airspeed gust bE/bh     : {r.bE_bh_gust:.4f}")
    print(f"    rear engine-out bE/bh   : {r.bE_bh_engine_out:.4f}")
    print(f"  Airspeed gust             : +/-{r.gust_speed:.2f} m/s")
    print(f"  Governing gust speed      : {r.gust_case_speed:.2f} m/s")
    print(f"  Gust tail moment demand   : {r.Cm_gust:+.5f}")
    print(f"  Rear engine-out Cm demand : {r.Cm_engine_out:+.5f}")
    print(f"  Stall-limited max bE/bh   : not applied")
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

    print("\n----- Tail Cruise Stall Check -----")
    print(f"  CLh (cruise, undeflected): {r.CLh_cruise:+.5f}")
    print(f"  Finite-tail CL_max       : {r.tail_CL_max_3d:+.5f}")
    print(f"  Allowed CL fraction      : {r.inputs.tail_clmax_fraction:.3f}")
    print(f"  Allowed |CLh| limit      : {r.tail_CL_limit:+.5f}")
    print(f"  CLh (elevator up max)    : {r.CLh_at_max_up:+.5f}  (diagnostic only)")
    print(f"  CLh (elevator down max)  : {r.CLh_at_max_down:+.5f}  (diagnostic only)")

    print("\n----- Elevator Geometry -----")
    print(f"  cE/ch                    : {g.cE_ch:.4f}")
    print(f"  bE/bh                    : {g.bE_bh:.4f}")
    print(f"  SE/Sh                    : {g.SE_Sh:.4f}")
    print(f"  tau_e                    : {r.tau_e:.4f}")