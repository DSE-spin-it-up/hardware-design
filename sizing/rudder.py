# Positive rudder deflection (rudder to the left) yaws nose to the left so Cn_deltaR is -ve.
"""Rudder sizing: minimum vertical-tail area for a configured rudder to hold a crosswind gust.

Procedure
---------
1.  Fix bR/bV and cR/cV (designer inputs) → derive τ_r from the empirical polynomial.
2.  Bisect on Sv to find the minimum vertical-tail area such that the configured
    rudder (bR/bV, cR/cV, δ_R_max) can simultaneously satisfy:
        a. Crosswind side-force balance
        b. Combined yaw-moment balance, including crosswind, Cn_dist, and OEI
3.  Recompute all tail geometry (bv, CLα_v, Vv, Sv/S, dc, Ss) at the minimum Sv.
4.  Solve for weathercock sideslip σ at the minimum Sv and configured rudder.
5.  Compute all stability/control derivatives and rudder hinge moment at cruise q
    and max deflection for boom torsion sizing.
"""
from dataclasses import dataclass, field

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
    bR_bV:              float = 1.00   # rudder span  / vertical-tail span  [-] (designer choice)
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
    active_constraint: str     # "combined_force" or "combined_yaw_moment"
    bR_bV_required:    float   # minimum bR/bV required by constraints [-]
    Sv_required:       float   # minimum vertical-tail area            [m²]
    force_margin:      float   # available minus required lateral force [N]
    moment_margin:     float   # available minus required yaw moment    [N·m]
    oei_moment:         float   # failed-prop yawing moment              [N·m]
    oei_moment_margin:  float   # rudder yaw-moment margin for OEI       [N·m]
    combined_yaw_moment: float    # total yawing moment requirement       [N·m]
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
    combined_yaw_moment: float = 0.0
    Cnb:                float = 0.0


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
    SR_SV       = i.cR_cV * i.bR_bV

    oei_failed_thrust = (
        (s.D if total_thrust is None else float(total_thrust))
        * i.oei_failed_thrust_fraction
    )
    oei_moment = abs(oei_failed_thrust * i.front_prop_lateral_position)

    gust_yaw_moment = abs(Fw * dc)
    dist_yaw_moment = abs(i.Cn_dist) * q_total * S * b
    combined_yaw_moment = (
        gust_yaw_moment
        + dist_yaw_moment
        + oei_moment
    )

    # Required non-dimensional side-force and yaw-moment coefficients. The
    # yaw demand is deliberately summed: crosswind and OEI are treated as
    # simultaneous design loads.
    cy_required = Fw / (q_total * S)
    cn_required = combined_yaw_moment / (q_total * S * b)

    # Fin contribution at full gust sideslip β
    cy_fin = i.Kf2 * CLalphav * i.eta_v * Sv / S * beta
    cn_fin = i.Kf1 * CLalphav * i.eta_v * Vv * beta

    # Non-dimensional yaw-stiffness derivative Cn_β for this Sv
    Cnb_value = i.Kf1 * CLalphav * i.eta_v * Sv * s.lh / (b * S)

    # Rudder contribution per unit (bR/bV) and per unit (bR/bV)²
    cy_rudder_per_b2 = CLalphav * i.eta_v * (Sv / S) * tau_r * i.cR_cV * delta_R_max
    cn_rudder_per_b  = CLalphav * Vv * i.eta_v * tau_r * delta_R_max

    # Minimum bR/bV from each constraint
    bR_bV_force = (
        float("inf") if cy_rudder_per_b2 <= 0.0
        else np.sqrt(max(cy_required - cy_fin, 0.0) / cy_rudder_per_b2)
    )
    bR_bV_moment = (
        float("inf") if cn_rudder_per_b <= 0.0
        else max(cn_required - cn_fin, 0.0) / cn_rudder_per_b
    )
    bR_bV_required = max(float(bR_bV_force), float(bR_bV_moment))

    # Margins at the configured bR/bV
    bR_bV_check    = min(max(i.bR_bV, 0.0), 1.0)
    cy_available   = cy_fin + cy_rudder_per_b2 * bR_bV_check ** 2
    cn_available   = cn_fin + cn_rudder_per_b  * bR_bV_check
    cn_rudder_available = cn_rudder_per_b * bR_bV_check
    force_margin   = q_total * S     * (cy_available - cy_required)
    moment_margin  = q_total * S * b * (cn_available - cn_required)
    oei_moment_margin = q_total * S * b * cn_rudder_available - oei_moment

    active = "combined_yaw_moment" if bR_bV_moment >= bR_bV_force else "combined_force"

    return RudderAreaRequirement(
        Sv                = Sv,
        bR_bV_required    = bR_bV_required,
        active_constraint = active,
        force_margin      = float(force_margin),
        moment_margin     = float(moment_margin),
        oei_moment        = float(oei_moment),
        oei_moment_margin = float(oei_moment_margin),
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
) -> RudderAreaRequirement:
    """
    Smallest Sv such that the configured rudder (bR/bV, cR/cV, δ_R_max)
    can meet both the side-force and yaw-moment constraints.
    """
    if inputs is None:
        inputs = RudderInputs()
    if not (0.0 < inputs.bR_bV <= 1.0):
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

    # Expand upper bound until the configured rudder is sufficient
    for _ in range(60):
        req_hi = _requirement_for_area(
            sizing, fus, v_stall, inputs, x_cg, hi, total_thrust,
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
    )
    for _ in range(80):
        mid     = 0.5 * (lo + hi)
        req_mid = _requirement_for_area(
            sizing, fus, v_stall, inputs, x_cg, mid, total_thrust,
        )
        if req_mid.bR_bV_required <= inputs.bR_bV and req_mid.Cnb >= inputs.Cn_beta:
            hi   = mid
            best = req_mid
        else:
            lo = mid

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
) -> RudderResult:
    """
    Size the vertical tail by finding the minimum Sv such that the
    configured rudder (bR/bV, cR/cV, δ_R_max) can hold the crosswind gust
    and counteract Cn_dist.  All derivatives are then computed at that Sv.

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
    )
    Sv_min   = area_req.Sv

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

    SR_SV = i.cR_cV * i.bR_bV
    Cndr  = -CLalphav * Vv             * i.eta_v * tau_r * i.bR_bV
    Cydr  =  CLalphav * i.eta_v * tau_r * i.bR_bV * SR_SV

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
    S_rudder = i.cR_cV * i.bR_bV * Sv_min
    c_rudder = i.cR_cV * c_v

    geometry = RudderGeometry(
        bR_bV    = i.bR_bV,
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
        combined_yaw_moment = area_req.combined_yaw_moment,
        hinge_moment      = hm,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summary(r: RudderResult) -> None:
    g  = r.geometry
    i  = r.inputs
    hm = r.hinge_moment
    Vv     = abs(r.Cndr) / (r.CLalphav * i.eta_v * r.tau_r * g.bR_bV)
    Sv_S   = -r.Cyb / (i.Kf2 * r.CLalphav * i.eta_v)
    lv_b   = Vv / Sv_S if Sv_S > 0.0 else float("nan")

    print("  Vertical tail (sized by rudder)")
    print(f"    Sv_required               : {g.Sv:.4f}  m^2")
    print(f"    bv                        : {g.bv:.4f}  m")
    print("  Rudder geometry")
    print(f"    cR/cV                     : {g.cR_cV:.4f}  (designer input)")
    print(f"    bR/bV                     : {g.bR_bV:.4f}  (designer input)")
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
    print(f"  VT lateral force margin     : {r.force_margin:+.2f}  N")
    print(f"  VT yaw moment margin        : {r.moment_margin:+.2f}  N*m")
    print(f"  OEI failed-thrust fraction  : {i.oei_failed_thrust_fraction:.4f}")
    print(f"  OEI front prop lateral arm  : {i.front_prop_lateral_position:.4f}  m")
    print(f"  OEI yaw moment              : {r.oei_moment:.2f}  N*m")
    print(f"  OEI yaw moment margin       : {r.oei_moment_margin:+.2f}  N*m")
    print(f"  Combined yaw moment demand  : {r.combined_yaw_moment:.2f}  N*m")
    print("  Hinge moment (cruise, delta_max)")
    print(f"    Chi                       : {hm.Chi:.4f}")
    print(f"    H                         : {hm.H:.4f}  N*m  (boom torsion input)")
