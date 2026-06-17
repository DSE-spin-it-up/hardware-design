# Positive rudder deflection (rudder to the left) yaws nose to the left so Cn_deltaR is -ve.
"""Rudder sizing: minimum vertical-tail area for a configured rudder to hold a crosswind gust.

Procedure
---------
1.  Fix bR/bV and cR/cV (designer inputs) → derive τ_r from the empirical polynomial.
2.  Bisect on Sv to find the minimum vertical-tail area such that the configured
    rudder (bR/bV, cR/cV, δ_R_max) can simultaneously satisfy:
        a. Crosswind side-force balance
        b. Combined yaw-moment balance, including crosswind, Cn_dist, and OEI
        c. Minimum weathercock stiffness Cn_beta
3.  Recompute all tail geometry (bv, CLα_v, Vv, Sv/S, dc, Ss) at the minimum Sv.
4.  Solve for weathercock sideslip σ at the minimum Sv and configured rudder.
5.  Compute all stability/control derivatives and rudder hinge moment at cruise q
    and max deflection for boom torsion sizing.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import fsolve

from aerodynamics.hinge_moment import (
    HingeMomentInputs,
    HingeMomentResult,
    compute as compute_hinge_moment,
)
from pipeline.helpers import ScissorData
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RudderInputs:
    """Fixed design parameters — set by the designer, never overwritten."""
    bR_bV:              float | None = 1.00  # rudder span / vertical-tail span [-]; None -> solve it
    cR_cV:              float = 0.40   # rudder chord / vertical-tail chord [-] (designer choice)
    max_deflection_deg: float = 30.0   # hard deflection limit [deg]
    Cn_dist:            float = 0.10   # max yaw-moment disturbance the rudder must
                                       # counteract at δ_R_max [-]
    Cn_beta:            float = 0.0    # required yaw-stiffness derivative Cn_β [1/rad]
    CDY:                float = 0.8    # fuselage side-drag coefficient [-]
    Kf1:                float = 0.85   # fuselage correction on Cn_β [-]
    Kf2:                float = 1.0    # fuselage correction on Cy_β [-]
    eta_v:              float = 0.85   # dynamic-pressure ratio at v-tail (Vv/V)² [-]
    front_prop_lateral_position: float = 0.0  # [m] lateral arm of the failed front prop
    oei_failed_thrust_fraction: float = 1.0 / 3.0  # [-] failed thrust / total drone thrust
    Ss:                 float | None = None
    # body side area [m²]
    # None → fus.length * fus.height + Sv  (box fuselage + vertical tail)
    hinge_moment: HingeMomentInputs = field(default_factory=HingeMomentInputs)
    # hinge-moment coefficients for the rudder surface


@dataclass
class RudderGeometry:
    """Derived geometry — output of sizing, never a designer input."""
    bR_bV:    float   # rudder span  / vertical-tail span  [-]
    cR_cV:    float   # rudder chord / vertical-tail chord [-]
    SR_SV:    float   # rudder area  / vertical-tail area  [-]
    S_rudder: float   # rudder planform area [m²]
    c_rudder: float   # rudder chord [m]
    Sv:       float   # minimum required vertical-tail area [m²]
    bv:       float   # vertical-tail span at minimum Sv [m]


@dataclass
class RudderResult:
    inputs:            RudderInputs
    geometry:          RudderGeometry
    CLalphav:          float   # vertical-tail lift-curve slope        [1/rad]
    tau_r:             float   # rudder effectiveness τ_r              [-]
    Cnb:               float   # yaw-stiffness derivative Cn_β         [1/rad]
    Cyb:               float   # side-force/sideslip derivative Cy_β   [1/rad]
    Cndr:              float   # yaw-control derivative Cn_δr          [1/rad]
    Cydr:              float   # side-force control derivative Cy_δr   [1/rad]
    beta_gust:         float   # geometric gust sideslip β             [rad]
    sigma:             float   # weathercock sideslip σ                [rad]
    delta_R:           float   # applied rudder deflection (= limit)   [rad]
    active_constraint: str     # combined_force, combined_yaw_moment, payload_yaw_moment, or cn_beta
    bR_bV_required:    float   # minimum bR/bV required by constraints [-]
    Sv_required:       float   # minimum vertical-tail area            [m²]
    force_margin:      float   # available minus required lateral force [N]
    moment_margin:     float   # available minus required yaw moment    [N·m]
    oei_moment:         float   # failed-prop yawing moment              [N·m]
    oei_moment_margin:  float   # rudder yaw-moment margin for OEI       [N·m]
    payload_yaw_moment: float   # payload yawing moment                   [N·m]
    payload_yaw_moment_margin: float  # rudder yaw-moment margin for payload [N·m]
    payload_attachment_dx: float  # CG to payload attachment x-offset       [m]
    combined_yaw_moment: float    # total yawing moment requirement       [N·m]
    cnbeta_margin:     float   # Cn_beta achieved minus Cn_beta required [1/rad]
    hinge_moment:      HingeMomentResult  # rudder hinge moment at cruise, δ_R_max


@dataclass
class RudderAreaRequirement:
    Sv:                float
    bR_bV_required:    float
    active_constraint: str
    force_margin:      float = 0.0
    moment_margin:     float = 0.0
    oei_moment:         float = 0.0
    oei_moment_margin:  float = 0.0
    payload_yaw_moment: float = 0.0
    payload_yaw_moment_margin: float = 0.0
    payload_attachment_dx: float = 0.0
    combined_yaw_moment: float = 0.0
    Cnb:                float = 0.0
    cnbeta_margin:      float = 0.0   # Cnb - Cn_beta_required


# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationship
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """
    Control-surface effectiveness τ from flap-chord ratio cf/c.
    Empirical 4th-order polynomial fit, valid for cf/c ∈ [0.15, 0.40].
    """
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


# ---------------------------------------------------------------------------
# Tail geometry helpers
# ---------------------------------------------------------------------------

def _vertical_tail_geometry_for_area(
    sizing: SizingResult, Sv: float
) -> tuple[float, float, float]:
    """Return (bv, CLalphav, Vv) for a given vertical-tail area Sv."""
    bv       = np.sqrt(2.0 * sizing.inputs.ARt * Sv) / 2.0
    ARv      = bv ** 2 / Sv
    CLalphav = 2.0 * np.pi * ARv / (ARv + 2.0)
    Vv       = Sv * sizing.lh / (sizing.Sw * sizing.inputs.b)
    return bv, CLalphav, Vv


def _payload_yaw_moment(
    sizing: SizingResult,
    y_cg: dict[str, float] | None,
    payload_max_tension: float | None,
) -> tuple[float, float]:
    """Return (yaw moment, x offset) for max payload tension at cruise angle."""
    if y_cg is None or payload_max_tension is None:
        return 0.0, 0.0
    if payload_max_tension < 0.0:
        raise ValueError("payload_max_tension must be non-negative.")

    n_drones = max(float(sizing.inputs.n_drones), 1.0)
    payload_weight = sizing.inputs.m_payload / n_drones * 9.80665
    payload_drag = (
        sizing.q_cruise
        * sizing.inputs.Cd_payload
        * sizing.inputs.S_payload
        / n_drones
    )
    cable_angle = float(np.arctan2(payload_drag, payload_weight))
    attachment_y = y_cg.get("pvc_tubes", y_cg.get("pvc_tube_bottom", y_cg.get("overall", 0.0)))
    vertical_drop = max(float(y_cg.get("overall", 0.0)) - float(attachment_y), 0.0)
    attachment_dx = vertical_drop * np.tan(cable_angle)
    return float(payload_max_tension * abs(attachment_dx)), float(attachment_dx)


# ---------------------------------------------------------------------------
# Requirement at a given Sv
# ---------------------------------------------------------------------------

def _requirement_for_area(
    sizing:  SizingResult,
    fus:     FuselageResult,
    v_stall: float,
    inputs:  RudderInputs,
    x_cg:    float,
    Sv:      float,
    total_thrust: float | None = None,
    y_cg: dict[str, float] | None = None,
    payload_max_tension: float | None = None,
    credit_fin_for_required: bool = True,
) -> RudderAreaRequirement:
    """
    For a given vertical-tail area Sv, compute what bR/bV is needed and
    the force/moment margins at the configured bR/bV.
    """
    s = sizing
    i = inputs
    S = s.Sw
    b = s.inputs.b
    rho = s.rho

    bv, CLalphav, Vv = _vertical_tail_geometry_for_area(s, Sv)

    A_fus         = fus.length * fus.height
    x_fus_lemac   = fus.x_nose + fus.length / 2.0
    x_vtail_lemac = 0.25 * s.c + s.lh
    Ss            = i.Ss if i.Ss is not None else A_fus + Sv
    dc            = (A_fus * x_fus_lemac + Sv * x_vtail_lemac) / Ss - x_cg

    gust_speed = s.inputs.gust_speed
    VT         = np.sqrt(v_stall ** 2 + gust_speed ** 2)
    beta       = np.arctan(gust_speed / v_stall)
    q_total    = 0.5 * rho * VT ** 2
    q_gust     = 0.5 * rho * gust_speed ** 2
    Fw         = q_gust * Ss * i.CDY

    delta_R_max = np.radians(i.max_deflection_deg)
    tau_r       = tau_from_chord_ratio(i.cR_cV)
    oei_failed_thrust = (
        (s.D if total_thrust is None else float(total_thrust))
        * i.oei_failed_thrust_fraction
    )
    oei_moment = abs(oei_failed_thrust * i.front_prop_lateral_position)
    payload_yaw_moment, payload_attachment_dx = _payload_yaw_moment(
        s,
        y_cg,
        payload_max_tension,
    )

    gust_yaw_moment = abs(Fw * dc)
    dist_yaw_moment = abs(i.Cn_dist) * q_total * S * b
    combined_yaw_moment = (
        gust_yaw_moment
        + dist_yaw_moment
        + oei_moment
    )

    cy_required = Fw / (q_total * S)
    cn_required = combined_yaw_moment / (q_total * S * b)

    cy_fin = i.Kf2 * CLalphav * i.eta_v * Sv / S * beta
    cn_fin = i.Kf1 * CLalphav * i.eta_v * Vv * beta

    Cnb_value = i.Kf1 * CLalphav * i.eta_v * Sv * s.lh / (b * S)

    cy_rudder_per_b2 = CLalphav * i.eta_v * (Sv / S) * tau_r * i.cR_cV * delta_R_max
    cn_rudder_per_b  = CLalphav * Vv * i.eta_v * tau_r * delta_R_max

    cy_fin_for_required = cy_fin if credit_fin_for_required else 0.0
    cn_fin_for_required = cn_fin if credit_fin_for_required else 0.0

    bR_bV_force = (
        float("inf") if cy_rudder_per_b2 <= 0.0
        else np.sqrt(max(cy_required - cy_fin_for_required, 0.0) / cy_rudder_per_b2)
    )
    bR_bV_moment = (
        float("inf") if cn_rudder_per_b <= 0.0
        else max(cn_required - cn_fin_for_required, 0.0) / cn_rudder_per_b
    )
    payload_yaw_moment_coeff = (
        payload_yaw_moment / (s.q_cruise * S * b)
        if s.q_cruise * S * b > 0.0 else float("inf")
    )
    if payload_yaw_moment <= 0.0:
        bR_bV_payload = 0.0
    else:
        bR_bV_payload = (
            float("inf") if cn_rudder_per_b <= 0.0
            else payload_yaw_moment_coeff / cn_rudder_per_b
        )
    bR_bV_required = max(
        float(bR_bV_force),
        float(bR_bV_moment),
        float(bR_bV_payload),
    )

    bR_bV_input    = 1.0 if i.bR_bV is None else i.bR_bV
    bR_bV_check    = min(max(bR_bV_input, 0.0), 1.0)
    cy_available   = cy_fin + cy_rudder_per_b2 * bR_bV_check ** 2
    cn_available   = cn_fin + cn_rudder_per_b  * bR_bV_check
    cn_rudder_available = cn_rudder_per_b * bR_bV_check
    force_margin   = q_total * S     * (cy_available - cy_required)
    moment_margin  = q_total * S * b * (cn_available - cn_required)
    oei_moment_margin = q_total * S * b * cn_rudder_available - oei_moment
    payload_yaw_moment_margin = (
        s.q_cruise * S * b * cn_rudder_available - payload_yaw_moment
    )

    constraint_requirements = {
        "combined_force": float(bR_bV_force),
        "combined_yaw_moment": float(bR_bV_moment),
        "payload_yaw_moment": float(bR_bV_payload),
    }
    active = max(constraint_requirements, key=constraint_requirements.get)

    return RudderAreaRequirement(
        Sv                = Sv,
        bR_bV_required    = bR_bV_required,
        active_constraint = active,
        force_margin      = float(force_margin),
        moment_margin     = float(moment_margin),
        oei_moment        = float(oei_moment),
        oei_moment_margin = float(oei_moment_margin),
        payload_yaw_moment = float(payload_yaw_moment),
        payload_yaw_moment_margin = float(payload_yaw_moment_margin),
        payload_attachment_dx = float(payload_attachment_dx),
        combined_yaw_moment = float(combined_yaw_moment),
        Cnb               = float(Cnb_value),
    )


# ---------------------------------------------------------------------------
# Minimum vertical-tail area bisection
# ---------------------------------------------------------------------------

def minimum_vertical_tail_area(
    sizing:  SizingResult,
    fus:     FuselageResult,
    v_stall: float,
    inputs:  RudderInputs | None = None,
    x_cg:    float = 0.0,
    total_thrust: float | None = None,
    y_cg: dict[str, float] | None = None,
    payload_max_tension: float | None = None,
) -> RudderAreaRequirement:
    """
    Smallest Sv such that the configured rudder (bR/bV, cR/cV, δ_R_max)
    can meet the side-force, yaw-moment, and Cn_beta constraints.
    """
    if inputs is None:
        inputs = RudderInputs()
    if inputs.bR_bV is not None and not (0.0 < inputs.bR_bV <= 1.0):
        raise ValueError("rudder.bR_bV must satisfy 0 < bR_bV <= 1.")
    if not (0.0 < inputs.cR_cV < 1.0):
        raise ValueError("rudder.cR_cV must satisfy 0 < cR_cV < 1.")
    if inputs.max_deflection_deg <= 0.0:
        raise ValueError("rudder.max_deflection_deg must be positive.")
    if inputs.front_prop_lateral_position < 0.0:
        raise ValueError("rudder.front_prop_lateral_position must be non-negative.")
    if not (0.0 <= inputs.oei_failed_thrust_fraction <= 1.0):
        raise ValueError("rudder.oei_failed_thrust_fraction must satisfy 0 <= value <= 1.")
    if inputs.Cn_beta < 0.0:
        raise ValueError("rudder.Cn_beta must be non-negative.")
    lo = max(1.0e-6, 1.0e-6 * sizing.Sw)
    hi = max(sizing.Sv, lo * 2.0)

    if inputs.bR_bV is None:
        # Auto-span mode: keep vertical-tail area tied only to Cn_beta, then
        # solve the rudder span ratio needed for control authority at that Sv.
        # Do not let passive fin stability reduce the solved rudder span to zero.
        for _ in range(60):
            req_hi = _requirement_for_area(
                sizing, fus, v_stall, inputs, x_cg, hi, total_thrust,
                y_cg=y_cg, payload_max_tension=payload_max_tension,
                credit_fin_for_required=False,
            )
            if req_hi.Cnb >= inputs.Cn_beta:
                break
            hi *= 2.0
        else:
            raise ValueError(
                "Could not find a vertical-tail area that satisfies "
                f"Cn_beta={inputs.Cn_beta:.3f}."
            )

        best = _requirement_for_area(
            sizing, fus, v_stall, inputs, x_cg, hi, total_thrust,
            y_cg=y_cg, payload_max_tension=payload_max_tension,
            credit_fin_for_required=False,
        )
        for _ in range(80):
            mid     = 0.5 * (lo + hi)
            req_mid = _requirement_for_area(
                sizing, fus, v_stall, inputs, x_cg, mid, total_thrust,
                y_cg=y_cg, payload_max_tension=payload_max_tension,
                credit_fin_for_required=False,
            )
            if req_mid.Cnb >= inputs.Cn_beta:
                hi   = mid
                best = req_mid
            else:
                lo = mid

        if best.bR_bV_required > 1.0 + 1.0e-9:
            raise ValueError(
                "Cn_beta-only vertical-tail area is too small for the rudder "
                "requirements: required bR/bV="
                f"{best.bR_bV_required:.3f} > 1.000. Increase Cn_beta, "
                "rudder chord ratio, or max deflection."
            )

        bR_bV_solved = min(max(best.bR_bV_required, 0.0), 1.0)
        best = _requirement_for_area(
            sizing,
            fus,
            v_stall,
            replace(inputs, bR_bV=bR_bV_solved),
            x_cg,
            best.Sv,
            total_thrust,
            y_cg=y_cg,
            payload_max_tension=payload_max_tension,
            credit_fin_for_required=False,
        )
        best.bR_bV_required = bR_bV_solved
        best.cnbeta_margin = best.Cnb - inputs.Cn_beta
        return best

    # Expand upper bound until the configured rudder is sufficient
    for _ in range(60):
        req_hi = _requirement_for_area(
            sizing, fus, v_stall, inputs, x_cg, hi, total_thrust,
            y_cg=y_cg, payload_max_tension=payload_max_tension,
        )
        if req_hi.bR_bV_required <= inputs.bR_bV and req_hi.Cnb >= inputs.Cn_beta:
            break
        hi *= 2.0
    else:
        raise ValueError(
            "Could not find a vertical-tail area that satisfies the rudder "
            f"constraints with bR/bV={inputs.bR_bV:.3f} and Cn_beta={inputs.Cn_beta:.3f}."
        )

    # Bisect to find the minimum Sv
    best = _requirement_for_area(
        sizing, fus, v_stall, inputs, x_cg, hi, total_thrust,
        y_cg=y_cg, payload_max_tension=payload_max_tension,
    )
    for _ in range(80):
        mid     = 0.5 * (lo + hi)
        req_mid = _requirement_for_area(
            sizing, fus, v_stall, inputs, x_cg, mid, total_thrust,
            y_cg=y_cg, payload_max_tension=payload_max_tension,
        )
        if req_mid.bR_bV_required <= inputs.bR_bV and req_mid.Cnb >= inputs.Cn_beta:
            hi   = mid
            best = req_mid
        else:
            lo = mid

    best.cnbeta_margin = best.Cnb - inputs.Cn_beta
    if best.Cnb < inputs.Cn_beta + 1e-9:
        best.active_constraint = "cn_beta"
    return best


# ---------------------------------------------------------------------------
# Main sizing routine
# ---------------------------------------------------------------------------

def run(
    sizing:  SizingResult,
    scissor: ScissorData,
    fus:     FuselageResult,
    v_stall: float,
    inputs:  RudderInputs | None = None,
    x_cg:    float = 0.0,
    total_thrust: float | None = None,
    y_cg: dict[str, float] | None = None,
    payload_max_tension: float | None = None,
) -> RudderResult:
    """
    Size the vertical tail by finding the minimum Sv such that the
    configured rudder (bR/bV, cR/cV, δ_R_max) can hold the crosswind gust,
    counteract Cn_dist, and achieve the required Cn_beta.  All derivatives
    are then computed at that Sv.

    Parameters
    ----------
    sizing  : converged wing/tail sizing result
    scissor : scissor-plot data (passed for pipeline consistency)
    fus     : fuselage geometry result
    v_stall : stall speed at the sizing condition [m/s]
    inputs  : RudderInputs overrides; defaults used when None
    x_cg    : aircraft CG position from LEMAC [m]  (from cg['overall'])

    Returns
    -------
    RudderResult with all derivatives, geometry, and hinge moment computed
    at the minimum required vertical-tail area.
    """
    if inputs is None:
        inputs = RudderInputs()
    i = inputs
    s = sizing

    # ------------------------------------------------------------------
    # 1. Find minimum Sv by bisection
    # ------------------------------------------------------------------
    area_req = minimum_vertical_tail_area(
        sizing, fus, v_stall, i, x_cg=x_cg, total_thrust=total_thrust,
        y_cg=y_cg, payload_max_tension=payload_max_tension,
    )
    Sv_min   = area_req.Sv
    bR_bV_design = area_req.bR_bV_required if i.bR_bV is None else i.bR_bV

    # ------------------------------------------------------------------
    # 2. Recompute all tail geometry at Sv_min
    # ------------------------------------------------------------------
    S  = s.Sw
    b  = s.inputs.b
    rho = s.rho
    lv  = s.lh

    bv, CLalphav, Vv = _vertical_tail_geometry_for_area(s, Sv_min)

    A_fus         = fus.length * fus.height
    x_fus_lemac   = fus.x_nose + fus.length / 2.0
    x_vtail_lemac = 0.25 * s.c + lv
    Ss            = i.Ss if i.Ss is not None else A_fus + Sv_min
    dc            = (A_fus * x_fus_lemac + Sv_min * x_vtail_lemac) / Ss - x_cg

    # ------------------------------------------------------------------
    # 3. Gust kinematics (independent of tail geometry)
    # ------------------------------------------------------------------
    gust_speed = s.inputs.gust_speed
    VT         = np.sqrt(v_stall ** 2 + gust_speed ** 2)
    beta       = np.arctan(gust_speed / v_stall)
    q_total    = 0.5 * rho * VT ** 2
    q_gust     = 0.5 * rho * gust_speed ** 2
    Fw         = q_gust * Ss * i.CDY

    delta_R_max = np.radians(i.max_deflection_deg)
    tau_r       = tau_from_chord_ratio(i.cR_cV)

    # ------------------------------------------------------------------
    # 4. Stability and control derivatives at Sv_min
    # ------------------------------------------------------------------
    Cyb  = -i.Kf2 * CLalphav * i.eta_v * Sv_min / S
    Cnb  =  i.Kf1 * CLalphav * i.eta_v * Sv_min * lv / (b * S)

    SR_SV = i.cR_cV * bR_bV_design
    Cndr  = -CLalphav * Vv             * i.eta_v * tau_r * bR_bV_design
    Cydr  =  CLalphav * i.eta_v * tau_r * bR_bV_design * SR_SV

    # ------------------------------------------------------------------
    # 5. Weathercock sideslip σ at Sv_min with configured rudder
    # ------------------------------------------------------------------
    def trim_sigma(sigma_only: list[float]) -> list[float]:
        sigma  = sigma_only[0]
        eq1    = (
            q_total * S * b * (Cnb * (beta - sigma) + Cndr * delta_R_max)
            + Fw * dc
        )
        return [eq1]

    sigma_final = float(fsolve(trim_sigma, [0.0])[0])

    # ------------------------------------------------------------------
    # 6. Rudder geometry at Sv_min
    # ------------------------------------------------------------------
    c_v      = Sv_min / bv
    S_rudder = i.cR_cV * bR_bV_design * Sv_min
    c_rudder = i.cR_cV * c_v

    geometry = RudderGeometry(
        bR_bV    = bR_bV_design,
        cR_cV    = i.cR_cV,
        SR_SV    = SR_SV,
        S_rudder = S_rudder,
        c_rudder = c_rudder,
        Sv       = Sv_min,
        bv       = bv,
    )

    # ------------------------------------------------------------------
    # 7. Hinge moment at cruise q and max deflection
    #    Critical case for boom torsion: max speed (cruise), max deflection.
    #    alpha used is the gust sideslip β (AoA seen by the vertical tail).
    # ------------------------------------------------------------------
    hm = compute_hinge_moment(
        alpha  = beta,
        delta  = delta_R_max,
        q      = s.q_cruise,
        S_ctrl = S_rudder,
        c_ctrl = c_rudder,
        inputs = i.hinge_moment,
    )

    return RudderResult(
        inputs            = inputs,
        geometry          = geometry,
        CLalphav          = CLalphav,
        tau_r             = tau_r,
        Cnb               = Cnb,
        Cyb               = Cyb,
        Cndr              = Cndr,
        Cydr              = Cydr,
        beta_gust         = beta,
        sigma             = sigma_final,
        delta_R           = delta_R_max,
        active_constraint = area_req.active_constraint,
        bR_bV_required    = area_req.bR_bV_required,
        Sv_required       = Sv_min,
        force_margin      = area_req.force_margin,
        moment_margin     = area_req.moment_margin,
        oei_moment        = area_req.oei_moment,
        oei_moment_margin = area_req.oei_moment_margin,
        payload_yaw_moment = area_req.payload_yaw_moment,
        payload_yaw_moment_margin = area_req.payload_yaw_moment_margin,
        payload_attachment_dx = area_req.payload_attachment_dx,
        combined_yaw_moment = area_req.combined_yaw_moment,
        cnbeta_margin     = area_req.cnbeta_margin,
        hinge_moment      = hm,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summary(r: RudderResult) -> None:
    g  = r.geometry
    i  = r.inputs
    hm = r.hinge_moment
    Vv     = r.Cnb / (i.Kf1 * r.CLalphav * i.eta_v)
    Sv_S   = -r.Cyb / (i.Kf2 * r.CLalphav * i.eta_v)
    lv_b   = Vv / Sv_S if Sv_S > 0.0 else float("nan")

    vt_basis = "Cn_beta" if i.bR_bV is None else "rudder"
    bR_basis = "solved" if i.bR_bV is None else "designer input"
    print(f"  Vertical tail (sized by {vt_basis})")
    print(f"    Sv_required               : {g.Sv:.4f}  m^2")
    print(f"    bv                        : {g.bv:.4f}  m")
    print("  Rudder geometry")
    print(f"    cR/cV                     : {g.cR_cV:.4f}  (designer input)")
    print(f"    bR/bV                     : {g.bR_bV:.4f}  ({bR_basis})")
    print(f"    required bR/bV            : {r.bR_bV_required:.4f}")
    print(f"    tau_r                     : {r.tau_r:.4f}")
    print(f"    SR/SV                     : {g.SR_SV:.4f}")
    print(f"    S_rudder                  : {g.S_rudder:.4f}  m^2")
    print(f"    c_rudder                  : {g.c_rudder:.4f}  m")
    print("  Sensitivity inputs:")
    print(f"    CL_alpha_v                : {r.CLalphav:.4f}  1/rad")
    print(f"    CL_alpha_v unit check     : {r.CLalphav * np.pi / 180.0:.5f}  1/deg")
    print(f"    eta_v                     : {i.eta_v:.4f}")
    print(f"    Vv = Sv*lv/(S*b)          : {Vv:.4f}")
    print(f"    Sv/S                      : {Sv_S:.4f}")
    print(f"    lv/b                      : {lv_b:.4f}")
    print(f"    bR/bV, SR/SV              : {g.bR_bV:.4f}, {g.SR_SV:.4f}")
    print(f"    delta_R                   : {np.degrees(r.delta_R):.2f}  deg")
    print("  Formulas:")
    print("    Cn_beta    =  Kf1*CL_alpha_v*eta_v*Sv/S*lv/b")
    print("    Cy_beta    = -Kf2*CL_alpha_v*eta_v*Sv/S")
    print("    Cn_delta_r = -CL_alpha_v*Vv*eta_v*tau_r*bR/bV")
    print("    Cy_delta_r =  CL_alpha_v*eta_v*tau_r*bR/bV*SR/SV")
    print(f"  Cn_beta                     : {r.Cnb:.4f}  1/rad")
    print(f"  Cy_beta                     : {r.Cyb:.4f}  1/rad")
    print(f"  Cn_delta_r                  : {r.Cndr:.4f}  1/rad")
    print(f"  Cy_delta_r                  : {r.Cydr:.4f}  1/rad")
    print(f"  Gust sideslip beta          : {np.degrees(r.beta_gust):.2f}  deg")
    print(f"  Weathercock sideslip sigma  : {np.degrees(r.sigma):.2f}  deg")
    print(f"  Applied rudder deflection   : {np.degrees(r.delta_R):+.2f}  deg  (= limit)")
    print(f"  Active sizing constraint    : {r.active_constraint}")
    print(f"  Cn_beta required            : {i.Cn_beta:.4f}  1/rad")
    print(f"  Cn_beta achieved            : {r.Cnb:.4f}  1/rad")
    print(f"  Cn_beta margin              : {r.cnbeta_margin:+.4f}  1/rad")
    print(f"  VT lateral force margin     : {r.force_margin:+.2f}  N")
    print(f"  VT yaw moment margin        : {r.moment_margin:+.2f}  N*m")
    print(f"  OEI failed-thrust fraction  : {i.oei_failed_thrust_fraction:.4f}")
    print(f"  OEI front prop lateral arm  : {i.front_prop_lateral_position:.4f}  m")
    print(f"  OEI yaw moment              : {r.oei_moment:.2f}  N*m")
    print(f"  OEI yaw moment margin       : {r.oei_moment_margin:+.2f}  N*m")
    print(f"  Payload attachment x offset : {r.payload_attachment_dx:.4f}  m")
    print(f"  Payload yaw moment          : {r.payload_yaw_moment:.2f}  N*m")
    print(f"  Payload yaw moment margin   : {r.payload_yaw_moment_margin:+.2f}  N*m")
    print(f"  Combined yaw moment demand  : {r.combined_yaw_moment:.2f}  N*m")
    print("  Hinge moment (cruise, delta_max)")
    print(f"    Chi                       : {hm.Chi:.4f}")
    print(f"    H                         : {hm.H:.4f}  N*m  (boom torsion input)")
