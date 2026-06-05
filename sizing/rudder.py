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
    cR_cV:              float = 0.30   # rudder chord / vertical-tail chord [-] (designer choice)
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
    # hinge-moment coefficients for the rudder surface


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
    hinge_moment:      HingeMomentResult  # rudder hinge moment at cruise, δ_R_max


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
    # Crosswind trim: solve (σ, bR/bV) simultaneously
    #
    # Control derivatives as functions of bR/bV:
    #   Cn_δr = -CLα_v · Vv · η_v · τ_r · (bR/bV)
    #   Cy_δr =  CLα_v · η_v · τ_r · (bR/bV) · (SR/SV)
    #          =  CLα_v · η_v · τ_r · (bR/bV) · (cR/cV · bR/bV)
    #          =  CLα_v · η_v · τ_r · (cR/cV) · (bR/bV)²
    #
    # Yaw moment balance (eq1):
    #   q·S·b · [Cn_β·(β-σ) + Cn_δr·δ_R] + Fw·dc = 0
    #
    # Side-force balance (eq2):
    #   q_gust·Ss·CDY = q_total·S · [Cy_β·(β-σ) + Cy_δr·δ_R]
    # ------------------------------------------------------------------
    def trim_eqs(x: list[float]) -> list[float]:
        sigma, bR_bV = x

        Cndr_t = -CLalphav * Vv      * i.eta_v * tau_r * bR_bV
        Cydr_t =  CLalphav * i.eta_v * tau_r   * i.cR_cV * bR_bV ** 2

        eq1 = (
            q_total * S * b * (Cnb * (beta - sigma) + Cndr_t * delta_R_max)
            + Fw * dc
        )
        eq2 = (
            q_gust * Ss * i.CDY
            - q_total * S * (Cyb * (beta - sigma) + Cydr_t * delta_R_max)
        )
        return [eq1, eq2]

    sol = fsolve(trim_eqs, [0.0, 0.5], full_output=True)
    x_sol, _, ier, msg = sol

    if ier != 1:
        raise ValueError(
            f"Rudder crosswind trim solver did not converge: {msg}\n"
            "Check gust speed, tail geometry, or deflection limit."
        )

    sigma_gust, bR_bV_gust = x_sol

    # ------------------------------------------------------------------
    # Cn_dist constraint: bR/bV required to counteract yaw disturbance
    #   |Cn_δr · δ_R_max| = Cn_dist
    #   CLα_v · Vv · η_v · τ_r · δ_R_max · bR/bV = Cn_dist
    # ------------------------------------------------------------------
    bR_bV_dist = i.Cn_dist / (CLalphav * Vv * i.eta_v * tau_r * delta_R_max)

    # ------------------------------------------------------------------
    # Active constraint
    # ------------------------------------------------------------------
    if bR_bV_gust >= bR_bV_dist:
        bR_bV_final       = bR_bV_gust
        sigma_final       = sigma_gust
        active_constraint = "gust"
    else:
        bR_bV_final       = bR_bV_dist
        # recompute σ at the larger bR/bV (gust trim with more rudder authority)
        def trim_sigma(sigma_only: list[float]) -> list[float]:
            sigma  = sigma_only[0]
            Cndr_t = -CLalphav * Vv * i.eta_v * tau_r * bR_bV_final
            eq1    = (
                q_total * S * b * (Cnb * (beta - sigma) + Cndr_t * delta_R_max)
                + Fw * dc
            )
            return [eq1]

        sigma_final       = float(fsolve(trim_sigma, [0.0])[0])
        active_constraint = "Cn_dist"

    if bR_bV_final > 1.0:
        raise ValueError(
            f"Rudder sizing failed: required bR/bV = {bR_bV_final:.3f} > 1.0.\n"
            f"Active constraint: {active_constraint}.\n"
            "Consider increasing cR/cV, increasing vertical tail size, or "
            "reducing the gust speed / Cn_dist requirement."
        )

    # ------------------------------------------------------------------
    # Final geometry and derivatives
    # ------------------------------------------------------------------
    c_v      = Sv / bv                          # mean vertical-tail chord [m]
    S_rudder = i.cR_cV * bR_bV_final * Sv      # rudder planform area [m²]
    c_rudder = i.cR_cV * c_v                    # rudder chord [m]

    geometry = RudderGeometry(
        bR_bV    = bR_bV_final,
        cR_cV    = i.cR_cV,
        SR_SV    = i.cR_cV * bR_bV_final,
        S_rudder = S_rudder,
        c_rudder = c_rudder,
    )

    Cndr = -CLalphav * Vv      * i.eta_v * tau_r * geometry.bR_bV
    Cydr =  CLalphav * i.eta_v * tau_r   * geometry.SR_SV

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
    print(f"    bR/bV                     : {g.bR_bV:.4f}  (from sizing)")
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
    print(f"    bR/bV                     : {g.bR_bV:.4f}  (from sizing)")
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
    print("    Cy_delta_r =  CL_alpha_v*eta_v*tau_r*SR/SV")
    print(f"  Cn_beta                     : {r.Cnb:.4f}  1/rad")
    print(f"  Cy_beta                     : {r.Cyb:.4f}  1/rad")
    print(f"  Cn_delta_r                  : {r.Cndr:.4f}  1/rad")
    print(f"  Cy_delta_r                  : {r.Cydr:.4f}  1/rad")
    print(f"  Gust sideslip beta          : {np.degrees(r.beta_gust):.2f}  deg")
    print(f"  Weathercock sideslip sigma  : {np.degrees(r.sigma):.2f}  deg")
    print(f"  Applied rudder deflection   : {np.degrees(r.delta_R):+.2f}  deg  (= limit)")
    print(f"  Active sizing constraint    : {r.active_constraint}")
    print("  Hinge moment (cruise, delta_max)")
    print(f"    Chi                       : {hm.Chi:.4f}")
    print(f"    H                         : {hm.H:.4f}  N*m  (boom torsion input)")
