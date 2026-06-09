# Positive rudder deflection (rudder to the left) yaws nose to the left so Cn_deltaR is -ve.
"""Rudder sizing: minimum rudder geometry to hold a crosswind gust.

Procedure
---------
1.  Compute vertical-tail aerodynamic properties (CLα_v, Vv) from the
    converged tail geometry.
2.  Fix cR/cV (designer input) → derive τ_r from the empirical polynomial.
3.  Solve the two simultaneous trim equations (yaw moment + side-force
    balance) for (σ, bR/bV) at δ_R = δ_R_max:
        Cn_β·(β-σ)·q·S·b + Cn_δr·δ_R·q·S·b + Fw·dc = 0   [yaw]
        Fw = q_gust·S·CDY - q_total·S·(Cy_β·(β-σ) + Cy_δr·δ_R)  [side-force]
4.  Compute bR/bV required to counteract Cn_dist alone at δ_R_max:
        bR/bV_dist = Cn_dist / (CLα_v · Vv · η_v · τ_r · δ_R_max)
5.  Active constraint is whichever gives the larger bR/bV.
6.  Derive SR/SV = (cR/cV) × (bR/bV)  (rectangular-panel assumption).
7.  Compute all derivatives and return.
8.  Compute rudder hinge moment at cruise q and max deflection for boom
    torsion sizing.
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
    bR_bV:              float = 1.00   # rudder span / vertical-tail span [-] (designer choice)
    cR_cV:              float = 0.40   # rudder chord / vertical-tail chord [-] (designer choice)
    max_deflection_deg: float = 30.0   # hard deflection limit [deg]
    Cn_dist:            float = 0.10   # max yaw-moment disturbance the rudder must
                                       # counteract at δ_R_max [-]
    CDY:                float = 0.8    # fuselage side-drag coefficient [-]
    Kf1:                float = 0.85   # fuselage correction on Cn_β [-]
    Kf2:                float = 1.0    # fuselage correction on Cy_β [-]
    eta_v:              float = 0.85   # dynamic-pressure ratio at v-tail (Vv/V)² [-]
    Ss:                 float | None = None
    # body side area [m²]
    # None → fus.length * fus.height + Sv  (box fuselage + vertical tail)
    hinge_moment: HingeMomentInputs = field(default_factory=HingeMomentInputs)




@dataclass
class RudderGeometry:
    """Derived geometry — output of sizing, never a designer input."""
    bR_bV: float   # rudder span  / vertical-tail span  [-]
    cR_cV: float   # rudder chord / vertical-tail chord [-]
    SR_SV: float   # rudder area  / vertical-tail area  [-]
    S_rudder: float  # rudder planform area [m²]
    c_rudder: float  # rudder chord [m]


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
    active_constraint: str     # "gust" or "Cn_dist"
    bR_bV_required:    float   # minimum rudder span ratio required by constraints [-]
    Sv_required:       float   # minimum vertical-tail area for configured rudder [m^2]
    force_margin:      float   # available minus required lateral force [N]
    moment_margin:     float   # available minus required yaw moment [N*m]
    hinge_moment:      HingeMomentResult  # rudder hinge moment at cruise, δ_R_max


@dataclass
class RudderAreaRequirement:
    Sv: float
    bR_bV_required: float
    active_constraint: str
    force_margin: float = 0.0
    moment_margin: float = 0.0


# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationship
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """
    Control-surface effectiveness τ from flap-chord ratio cf/c.
    Empirical 4th-order polynomial fit, valid for cf/c ∈ [0.15, 0.40].
    """
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


def _vertical_tail_geometry_for_area(sizing: SizingResult, Sv: float) -> tuple[float, float, float]:
    bv = np.sqrt(2.0 * sizing.inputs.ARt * Sv) / 2.0
    ARv = bv ** 2 / Sv
    CLalphav = 2.0 * np.pi * ARv / (ARv + 2.0)
    Vv = Sv * sizing.lh / (sizing.Sw * sizing.inputs.b)
    return bv, CLalphav, Vv


def _requirement_for_area(
    sizing: SizingResult,
    fus: FuselageResult,
    v_stall: float,
    inputs: RudderInputs,
    x_cg: float,
    Sv: float,
) -> RudderAreaRequirement:
    s = sizing
    i = inputs
    S = s.Sw
    b = s.inputs.b
    rho = s.rho
    _, CLalphav, Vv = _vertical_tail_geometry_for_area(s, Sv)

    A_fus = fus.length * fus.height
    x_fus_lemac = fus.x_nose + fus.length / 2.0
    x_vtail_lemac = 0.25 * s.c + s.lh
    Ss = i.Ss if i.Ss is not None else A_fus + Sv
    dc = (A_fus * x_fus_lemac + Sv * x_vtail_lemac) / Ss - x_cg

    gust_speed = s.inputs.gust_speed
    VT = np.sqrt(v_stall ** 2 + gust_speed ** 2)
    beta = np.arctan(gust_speed / v_stall)
    q_total = 0.5 * rho * VT ** 2
    q_gust = 0.5 * rho * gust_speed ** 2
    Fw = q_gust * Ss * i.CDY
    delta_R_max = np.radians(i.max_deflection_deg)
    tau_r = tau_from_chord_ratio(i.cR_cV)
    cy_required = Fw / (q_total * S)
    cn_required = abs(Fw * dc) / (q_total * S * b) + abs(i.Cn_dist)

    cy_fin = i.Kf2 * CLalphav * i.eta_v * Sv / S * beta
    cn_fin = i.Kf1 * CLalphav * i.eta_v * Vv * beta

    cy_rudder_per_b2 = CLalphav * i.eta_v * (Sv / S) * tau_r * i.cR_cV * delta_R_max
    cn_rudder_per_b = CLalphav * Vv * i.eta_v * tau_r * delta_R_max

    bR_bV_force = (
        float("inf") if cy_rudder_per_b2 <= 0.0
        else np.sqrt(max(cy_required - cy_fin, 0.0) / cy_rudder_per_b2)
    )
    bR_bV_moment = (
        float("inf") if cn_rudder_per_b <= 0.0
        else max(cn_required - cn_fin, 0.0) / cn_rudder_per_b
    )
    bR_bV_required = max(float(bR_bV_force), float(bR_bV_moment))

    bR_bV_check = min(max(i.bR_bV, 0.0), 1.0)
    cy_available = cy_fin + cy_rudder_per_b2 * bR_bV_check ** 2
    cn_available = cn_fin + cn_rudder_per_b * bR_bV_check
    force_margin = q_total * S * (cy_available - cy_required)
    moment_margin = q_total * S * b * (cn_available - cn_required)

    if bR_bV_moment >= bR_bV_force:
        active = "Cn_dist" if abs(i.Cn_dist) >= abs(Fw * dc) / (q_total * S * b) else "gust_moment"
    else:
        active = "gust_force"
    return RudderAreaRequirement(
        Sv=Sv,
        bR_bV_required=bR_bV_required,
        active_constraint=active,
        force_margin=float(force_margin),
        moment_margin=float(moment_margin),
    )


def minimum_vertical_tail_area(
    sizing: SizingResult,
    fus: FuselageResult,
    v_stall: float,
    inputs: RudderInputs | None = None,
    x_cg: float = 0.0,
) -> RudderAreaRequirement:
    """Smallest Sv that lets the configured rudder span/chord meet rudder constraints."""
    if inputs is None:
        inputs = RudderInputs()
    if inputs.bR_bV <= 0.0 or inputs.bR_bV > 1.0:
        raise ValueError("rudder.bR_bV must satisfy 0 < bR_bV <= 1.")
    if inputs.cR_cV <= 0.0 or inputs.cR_cV >= 1.0:
        raise ValueError("rudder.cR_cV must satisfy 0 < cR_cV < 1.")
    if inputs.max_deflection_deg <= 0.0:
        raise ValueError("rudder.max_deflection_deg must be positive.")

    def requirement(Sv: float) -> RudderAreaRequirement:
        return _requirement_for_area(sizing, fus, v_stall, inputs, x_cg, Sv)

    lo = max(1.0e-6, 1.0e-6 * sizing.Sw)
    hi = max(sizing.Sv, lo * 2.0)
    req_hi = requirement(hi)
    for _ in range(60):
        if req_hi.bR_bV_required <= inputs.bR_bV:
            break
        hi *= 2.0
        req_hi = requirement(hi)
    else:
        raise ValueError(
            "Could not find a vertical-tail area that satisfies the rudder "
            f"constraints with bR/bV={inputs.bR_bV:.3f}."
        )

    best = req_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        req_mid = requirement(mid)
        if req_mid.bR_bV_required <= inputs.bR_bV:
            hi = mid
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
) -> RudderResult:
    """
    Size the rudder by:
      1. Fixing cR/cV (designer input) → τ_r.
      2. Solving (σ, bR/bV) from the crosswind trim equations at δ_R_max.
      3. Computing bR/bV required to counteract Cn_dist alone.
      4. Taking the larger of the two as the active constraint.
      5. Computing the rudder hinge moment at cruise q and δ_R_max for
         boom torsion sizing.

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
    RudderResult with all derivatives, geometry, and hinge moment.
    """
    if inputs is None:
        inputs = RudderInputs()
    i = inputs
    s = sizing
    area_requirement = minimum_vertical_tail_area(
        sizing, fus, v_stall, i, x_cg=x_cg,
    )
    current_requirement = _requirement_for_area(
        sizing, fus, v_stall, i, x_cg, sizing.Sv,
    )

    # ------------------------------------------------------------------
    # Aircraft / tail geometry
    # ------------------------------------------------------------------
    rho = s.rho
    S   = s.Sw
    b   = s.inputs.b
    Sv  = s.Sv
    bv  = s.bv
    lv  = s.lh

    A_fus          = fus.length * fus.height
    x_fus_lemac    = fus.x_nose + fus.length / 2.0   # fuselage centroid from LEMAC
    x_vtail_lemac  = 0.25 * s.c + s.lh               # v-tail AC from LEMAC

    Ss = i.Ss if i.Ss is not None else A_fus + Sv
    dc = (A_fus * x_fus_lemac + Sv * x_vtail_lemac) / Ss - x_cg

    # ------------------------------------------------------------------
    # Vertical-tail aerodynamics (independent of rudder geometry)
    # ------------------------------------------------------------------
    ARv      = bv ** 2 / Sv
    CLalphav = 2 * np.pi * ARv / (ARv + 2)   # Helmbold low-speed slope [1/rad]
    Vv       = Sv * lv / (S * b)              # vertical-tail volume coefficient

    Cyb = -i.Kf2 * CLalphav * i.eta_v * Sv / S
    Cnb =  i.Kf1 * CLalphav * i.eta_v * Sv * lv / (b * S)

    # ------------------------------------------------------------------
    # Gust kinematics
    # ------------------------------------------------------------------
    gust_speed = s.inputs.gust_speed
    VT      = np.sqrt(v_stall ** 2 + gust_speed ** 2)   # total speed in gust [m/s]
    beta    = np.arctan(gust_speed / v_stall)            # gust sideslip [rad]
    q_total = 0.5 * rho * VT ** 2
    q_gust  = 0.5 * rho * gust_speed ** 2
    Fw      = q_gust * Ss * i.CDY                     # lateral gust force [N]

    delta_R_max = np.radians(i.max_deflection_deg)

    # ------------------------------------------------------------------
    # Rudder chord ratio → effectiveness (fixed by designer)
    # ------------------------------------------------------------------
    tau_r = tau_from_chord_ratio(i.cR_cV)

    # ------------------------------------------------------------------
    # Active constraint from explicit VT side-force and yaw-moment capacity.
    # The area loop does not rely on unbounded weathercock angle sigma.
    # ------------------------------------------------------------------
    bR_bV_final = current_requirement.bR_bV_required
    active_constraint = current_requirement.active_constraint

    def trim_sigma(sigma_only: list[float]) -> list[float]:
        sigma = sigma_only[0]
        Cndr_t = -CLalphav * Vv * i.eta_v * tau_r * min(bR_bV_final, i.bR_bV)
        eq1 = (
            q_total * S * b * (Cnb * (beta - sigma) + Cndr_t * delta_R_max)
            + Fw * dc
        )
        return [eq1]

    sigma_final = float(fsolve(trim_sigma, [0.0])[0])

    if bR_bV_final > i.bR_bV + 1.0e-9:
        raise ValueError(
            f"Rudder sizing failed: required bR/bV = {bR_bV_final:.3f} > configured bR/bV = {i.bR_bV:.3f}.\n"
            f"Active constraint: {active_constraint}.\n"
            f"Minimum vertical-tail area for this rudder is {area_requirement.Sv:.4f} m^2.\n"
            "Consider increasing cR/cV, increasing bR/bV, increasing vertical tail size, or "
            "reducing the gust speed / Cn_dist requirement."
        )

    # ------------------------------------------------------------------
    # Final geometry and derivatives
    # ------------------------------------------------------------------
    c_v      = Sv / bv                          # mean vertical-tail chord [m]
    S_rudder = i.cR_cV * i.bR_bV * Sv      # rudder planform area [m²]
    c_rudder = i.cR_cV * c_v                    # rudder chord [m]

    geometry = RudderGeometry(
        bR_bV    = i.bR_bV,
        cR_cV    = i.cR_cV,
        SR_SV    = i.cR_cV * i.bR_bV,
        S_rudder = S_rudder,
        c_rudder = c_rudder,
    )

    Cndr = -CLalphav * Vv      * i.eta_v * tau_r * geometry.bR_bV
    Cydr =  CLalphav * i.eta_v * (Sv / S) * tau_r * geometry.SR_SV

    # ------------------------------------------------------------------
    # Hinge moment at cruise q and max deflection
    # Critical case for boom torsion: max speed (cruise), max deflection.
    # alpha used is the gust sideslip β (AoA seen by the vertical tail).
    # ------------------------------------------------------------------
    hm = compute_hinge_moment(
        alpha   = beta,
        delta   = delta_R_max,
        q       = s.q_cruise,
        S_ctrl  = S_rudder,
        c_ctrl  = c_rudder,
        inputs  = i.hinge_moment,
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
        active_constraint = active_constraint,
        bR_bV_required    = bR_bV_final,
        Sv_required       = area_requirement.Sv,
        force_margin      = current_requirement.force_margin,
        moment_margin     = current_requirement.moment_margin,
        hinge_moment      = hm,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _summary_legacy(r: RudderResult) -> None:
    g = r.geometry
    i = r.inputs
    hm = r.hinge_moment
    print(f"  Rudder geometry")
    print(f"    cR/cV                     : {g.cR_cV:.4f}  (designer input)")
    print(f"    τ_r                       : {r.tau_r:.4f}")
    print(f"    bR/bV                     : {g.bR_bV:.4f}  (designer input)")
    print(f"    required bR/bV            : {r.bR_bV_required:.4f}")
    print(f"    required Sv               : {r.Sv_required:.4f}  m^2")
    print(f"    SR/SV                     : {g.SR_SV:.4f}")
    print(f"    S_rudder                  : {g.S_rudder:.4f}  m²")
    print(f"    c_rudder                  : {g.c_rudder:.4f}  m")
    print(f"  Vertical-tail CLα           : {r.CLalphav:.3f}  1/rad")
    print(f"  Cn_β                        : {r.Cnb:.4f}  1/rad")
    print(f"  Cy_β                        : {r.Cyb:.4f}  1/rad")
    print(f"  Cn_δr                       : {r.Cndr:.4f}  1/rad")
    print(f"  Cy_δr                       : {r.Cydr:.4f}  1/rad")
    print(f"  Gust sideslip β             : {np.degrees(r.beta_gust):.2f}  °")
    print(f"  Weathercock sideslip σ      : {np.degrees(r.sigma):.2f}  °")
    print(f"  Applied rudder deflection   : {np.degrees(r.delta_R):+.2f}  °  (= limit)")
    print(f"  Active sizing constraint    : {r.active_constraint}")
    print(f"  VT lateral force margin     : {r.force_margin:+.2f}  N")
    print(f"  VT yaw moment margin        : {r.moment_margin:+.2f}  N*m")
    print(f"  Hinge moment (cruise, δ_max)")
    print(f"    Chi                       : {hm.Chi:.4f}")
    print(f"    H                         : {hm.H:.4f}  N·m  (boom torsion input)")
def summary(r: RudderResult) -> None:
    g = r.geometry
    i = r.inputs
    hm = r.hinge_moment
    Vv = abs(r.Cndr) / (r.CLalphav * i.eta_v * r.tau_r * g.bR_bV)
    Sv_S = -r.Cyb / (i.Kf2 * r.CLalphav * i.eta_v)
    lv_b = Vv / Sv_S if Sv_S > 0.0 else float("nan")
    bR_bV_dist = i.Cn_dist / (r.CLalphav * Vv * i.eta_v * r.tau_r * r.delta_R)

    print("  Rudder geometry")
    print(f"    cR/cV                     : {g.cR_cV:.4f}  (designer input)")
    print(f"    tau_r                     : {r.tau_r:.4f}")
    print(f"    bR/bV                     : {g.bR_bV:.4f}  (designer input)")
    print(f"    required bR/bV            : {r.bR_bV_required:.4f}")
    print(f"    required Sv               : {r.Sv_required:.4f}  m^2")
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
    print(f"    bR/bV from Cn_dist        : {bR_bV_dist:.4f}")
    print("  Formulas:")
    print("    Cn_beta    =  Kf1*CL_alpha_v*eta_v*Sv/S*lv/b")
    print("    Cy_beta    = -Kf2*CL_alpha_v*eta_v*Sv/S")
    print("    Cn_delta_r = -CL_alpha_v*Vv*eta_v*tau_r*bR/bV")
    print("    Cy_delta_r =  CL_alpha_v*eta_v*Sv/S*tau_r*SR/SV")
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
    print("  Hinge moment (cruise, delta_max)")
    print(f"    Chi                       : {hm.Chi:.4f}")
    print(f"    H                         : {hm.H:.4f}  N*m  (boom torsion input)")
