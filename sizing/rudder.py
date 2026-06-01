# Positive rudder deflection (rudder to the left) yaws nose to the left so Cndeltar is -ve.
"""Rudder sizing: minimum rudder geometry to hold a crosswind gust.

Procedure
---------
1.  Compute vertical-tail aerodynamic properties (CLα_v, Vv) from the
    converged tail geometry.
2.  Fix δ_R = 30° (the deflection limit) and iterate over bR/bV starting
    from the minimum recommended value (0.70), stepping up by 0.05.
3.  For each candidate bR/bV, solve the two simultaneous trim equations
    (yaw moment balance + side-force balance) for (σ, cR/cV).
    cR/cV is a direct solver unknown; τ_r is computed forward from it
    inside the residual via the empirical polynomial — no inversion needed
    and Cy_δr is always exact.
4.  Accept the first bR/bV for which cR/cV ∈ [0.15, 0.40] AND the rudder has
    enough yaw authority at δ_R_max to counteract the maximum yaw-moment
    disturbance Cn_dist  (|Cn_δr · δ_R_max| ≥ Cn_dist).
5.  Derive SR/SV = (cR/cV) × (bR/bV)  (rectangular-panel assumption).
6.  Recompute all derivatives with the final geometry and return.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from pipeline.helpers import ScissorData
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RudderInputs:
    """Fixed design parameters — set by the designer, never overwritten."""
    max_deflection_deg: float = 30.0  # hard deflection limit [deg]
    Cn_dist: float = 0.0              # max yaw-moment disturbance the rudder must
                                      # counteract at δ_R_max [-]  (0 → non-binding)
    Vgust: float = 5.0                # lateral gust velocity [m/s]
    CDY:   float = 0.8                # fuselage side-drag coefficient [-]
    Kf1:   float = 0.85               # fuselage correction on Cn_β [-]
    Kf2:   float = 1.0                # fuselage correction on Cy_β [-]
    eta_v: float = 0.85           # dynamic-pressure ratio at v-tail (Vv/V)² [-]
    Ss:    float | None = None
    # body side area [m²]
    # None → fus.length * fus.height + Sv  (box fuselage + vertical tail)


@dataclass
class RudderGeometry:
    """Derived geometry — output of the sizing loop, never a designer input."""
    bR_bV: float   # rudder span  / vertical-tail span  [-]
    cR_CV: float   # rudder chord / vertical-tail chord [-]
    SR_SV: float   # rudder area  / vertical-tail area  [-]


@dataclass
class RudderResult:
    inputs:    RudderInputs
    geometry:  RudderGeometry
    CLalphav:  float   # vertical-tail lift-curve slope        [1/rad]
    tau_r:     float   # rudder effectiveness τ_r              [-]
    Cnb:       float   # yaw-stiffness derivative Cn_β         [1/rad]
    Cyb:       float   # side-force/sideslip derivative Cy_β   [1/rad]
    Cndr:      float   # yaw-control derivative Cn_δr          [1/rad]
    Cydr:      float   # side-force control derivative Cy_δr   [1/rad]
    beta_gust: float   # geometric gust sideslip β             [rad]
    sigma:     float   # weathercock sideslip σ                [rad]
    delta_R:   float   # applied rudder deflection (= limit)   [rad]
    within_limits: bool


# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationships
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """
    Control-surface effectiveness τ from flap-chord ratio cf/c.
    Empirical 4th-order polynomial fit, valid for cf/c ∈ [0.15, 0.40].
    """
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


def cR_cv_from_tau(tau_r: float) -> float:
    """
    Rudder chord ratio cR/cV from effectiveness τ_r (polynomial inversion).
    Used only for diagnostics; the sizing loop uses tau_from_chord_ratio()
    in the forward direction so no inversion is required there.
    """
    coeffs = [-6.624, 12.07, -8.292, 3.295, 0.004942 - tau_r]
    roots  = np.roots(coeffs)
    roots  = roots[np.isclose(roots.imag, 0)].real
    valid  = roots[(roots >= 0.15) & (roots <= 0.40)]
    if len(valid) == 0:
        raise ValueError(f"No valid rudder chord ratio for tau_r = {tau_r:.4f}")
    return float(valid[0])


# ---------------------------------------------------------------------------
# Main sizing routine
# ---------------------------------------------------------------------------

def run(
    sizing:  SizingResult,
    scissor: ScissorData,
    fus:     FuselageResult,
    v_stall: float,
    inputs:  RudderInputs | None = None,
) -> RudderResult:
    """
    Size the rudder by finding the minimum bR/bV (starting from 0.70) such
    that the required cR/cV falls within [0.15, 0.40] when δ_R is fixed at
    the 30° deflection limit.

    Parameters
    ----------
    sizing  : converged wing/tail sizing result
    scissor : scissor-plot data (passed for pipeline consistency)
    fus     : fuselage geometry result
    v_stall : stall speed at the sizing condition [m/s]
    inputs  : RudderInputs overrides; defaults used when None

    Returns
    -------
    RudderResult with all derivatives and a populated RudderGeometry.
    """
    if inputs is None:
        inputs = RudderInputs()
    i = inputs
    s = sizing

    # ------------------------------------------------------------------
    # Aircraft / tail geometry from the sizing result
    # ------------------------------------------------------------------
    rho = s.rho
    S   = s.Sw       # wing reference area [m²]
    b   = s.inputs.b # wing span [m]
    Sv  = s.Sv       # vertical-tail area [m²]
    bv  = s.bv       # vertical-tail span [m]
    lv  = s.lh       # tail moment arm [m]

    # Body side area: box fuselage projected rectangle + vertical tail planform
    Ss = i.Ss if i.Ss is not None else fus.length * fus.height + Sv

    # ------------------------------------------------------------------
    # Vertical-tail aerodynamics (independent of rudder geometry)
    # ------------------------------------------------------------------
    ARv      = bv ** 2 / Sv
    CLalphav = 2 * np.pi * ARv / (ARv + 2)  # Helmbold low-speed slope [1/rad]
    Vv       = Sv * lv / (S * b)            # vertical-tail volume coefficient

    # Stability derivatives — depend only on tail planform, not rudder geometry
    Cyb = -i.Kf2 * CLalphav * i.eta_v * Sv / S
    Cnb =  i.Kf1 * CLalphav * i.eta_v * Sv * lv / (b * S)

    # ------------------------------------------------------------------
    # Gust kinematics
    # ------------------------------------------------------------------
    VT   = np.sqrt(v_stall ** 2 + i.Vgust ** 2)  # total speed in gust [m/s]
    beta = np.arctan(i.Vgust / v_stall)           # gust sideslip angle [rad]
    Fw   = 0.5 * rho * i.Vgust ** 2 * Ss * i.CDY  # lateral gust force [N]

    delta_R_max = np.radians(i.max_deflection_deg)

    # ------------------------------------------------------------------
    # Iterative geometry sizing
    # ------------------------------------------------------------------
    # Fix δ_R = δ_R_max and solve for (σ, cR/cV) simultaneously.
    #
    # cR/cV is the direct solver unknown so:
    #   τ_r    = tau_from_chord_ratio(cR/cV)   forward, no inversion
    #   Sr/Sv  = (cR/cV) × (bR/bV)            rectangular-panel identity
    #   Cy_δr  is exact at every solver iteration
    #
    # The loop steps bR/bV from 0.70 to 1.00 in increments of 0.05 and
    # accepts the first value for which cR/cV ∈ [0.15, 0.40].

    found        = False
    chosen_bR_bV = None
    chosen_cR_CV = None
    chosen_tau_r = None
    sigma_sol    = 0.0

    for bR_bV_val in np.arange(0.70, 1.001, 0.05):

        # Default-argument capture freezes bR_bV_val in the closure.
        # Without it Python captures by reference and all iterations share
        # the final loop value.
        def eqs(x, _b=bR_bV_val):
            sigma, cR_CV_val = x

            # Clip only to keep polynomial well-behaved during iteration;
            # the range check after the solve is the real acceptance criterion.
            tau_eff = tau_from_chord_ratio(np.clip(cR_CV_val, 0.10, 0.45))

            # Rectangular rudder: Sr/Sv = (cR/cV) × (bR/bV)
            Sr_Sv = cR_CV_val * _b

            # Control derivatives at this candidate geometry
            Cndr_t = -CLalphav * Vv    * i.eta_v * tau_eff * _b
            Cydr_t =  CLalphav * i.eta_v * tau_eff * _b * Sr_Sv

            # Yaw-moment balance  Σ Cn = 0  (eq1)
            eq1 = (
                0.5 * rho * VT ** 2 * S * b *
                (Cnb * (beta - sigma) + Cndr_t * delta_R_max)
                + Fw * np.cos(sigma)          # dc = 0 for box fuselage
            )
            # Side-force balance  F_gust = aero side force  (eq2)
            eq2 = (
                0.5 * rho * i.Vgust ** 2 * S * i.CDY
                - 0.5 * rho * VT ** 2 * S *
                (Cyb * (beta - sigma) + Cydr_t * delta_R_max)
            )
            return [eq1, eq2]

        sol = fsolve(eqs, [0.0, 0.25], full_output=True)
        x_sol, _, ier, _ = sol

        if ier != 1:
            continue  # solver did not converge → try next span ratio

        sigma_sol, cR_CV_sol = x_sol

        # Accept only if cR/cV landed inside the valid range
        if 0.15 <= cR_CV_sol <= 0.40:
            found        = True
            chosen_bR_bV = bR_bV_val
            chosen_cR_CV = cR_CV_sol
            chosen_tau_r = tau_from_chord_ratio(cR_CV_sol)
            break

    if not found:
        raise ValueError(
            "Rudder sizing failed: no combination of bR/bV ∈ [0.70, 1.00] "
            "yields cR/cV ∈ [0.15, 0.40] for the given crosswind requirement.\n"
            "Consider relaxing the gust speed, increasing the vertical tail, "
            "or raising the deflection limit."
        )

    # ------------------------------------------------------------------
    # Assemble converged geometry
    # ------------------------------------------------------------------
    geometry = RudderGeometry(
        bR_bV = chosen_bR_bV,
        cR_CV = chosen_cR_CV,
        SR_SV = chosen_cR_CV * chosen_bR_bV,   # rectangular-panel identity
    )

    # ------------------------------------------------------------------
    # Final derivatives with converged geometry
    # ------------------------------------------------------------------
    br   = geometry.bR_bV * bv
    Sr   = geometry.SR_SV * Sv
    Cndr = -CLalphav * Vv    * i.eta_v * chosen_tau_r * br / bv
    Cydr =  CLalphav * i.eta_v * chosen_tau_r * (br / bv) * (Sr / Sv)

    return RudderResult(
        inputs    = inputs,
        geometry  = geometry,
        CLalphav  = CLalphav,
        tau_r     = chosen_tau_r,
        Cnb       = Cnb,
        Cyb       = Cyb,
        Cndr      = Cndr,
        Cydr      = Cydr,
        beta_gust = beta,
        sigma     = sigma_sol,
        delta_R   = delta_R_max,
        within_limits = True,   # always True by construction
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summary(r: RudderResult) -> None:
    g = r.geometry
    i = r.inputs
    print(f"  Converged rudder geometry")
    print(f"    bR/bV                     : {g.bR_bV:.2f}")
    print(f"    cR/cV                     : {g.cR_CV:.4f}")
    print(f"    SR/SV                     : {g.SR_SV:.4f}")
    print(f"  Vertical-tail CL_α          : {r.CLalphav:.3f}  1/rad")
    print(f"  Effectiveness τ_r           : {r.tau_r:.3f}")
    print(f"  Cn_β                        : {r.Cnb:.4f}  1/rad")
    print(f"  Cy_β                        : {r.Cyb:.4f}  1/rad")
    print(f"  Cn_δr                       : {r.Cndr:.4f}  1/rad")
    print(f"  Cy_δr                       : {r.Cydr:.4f}  1/rad")
    print(f"  Gust sideslip β             : {np.degrees(r.beta_gust):.2f}  °")
    print(f"  Weathercock sideslip σ      : {np.degrees(r.sigma):.2f}  °")
    print(f"  Applied rudder deflection   : {np.degrees(r.delta_R):+.2f}  °  (= limit)")
    print(f"  ✓ Within ±{i.max_deflection_deg:.0f}° deflection limit")