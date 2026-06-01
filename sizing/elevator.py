# Positive elevator deflection (elevator down) produces nose-down pitch so CM_deltaE is -ve.
"""Elevator sizing: minimum elevator geometry to trim at cruise.

Procedure
---------
1.  Compute tail angle of attack at cruise from downwash and incidence.
2.  Solve the pitch-moment-disturbance requirement analytically for τ_e
    (fixing δ_e = δ_up_max): the elevator-induced pitching moment at full
    up-deflection must counteract the maximum cruise pitch disturbance
    Cm_dist.  Because the elevator moment is written through the tail volume
    (ΔCm = -CLαh·τ_e·δ·η_h·Vh), this is independent of bE/bh.  Invert τ_e to
    get cE/ch.
3.  Iterate bE/bh from the minimum (0.03) upward in steps of 0.01.
    For each candidate:
        a.  Check cE/ch ∈ [0.20, 0.40]  (same for every iteration, but
            validated here so the loop can raise a clean error if it fails).
        b.  Compute CM_δe and CL_δe with the candidate bE/bh.
        c.  Compute the cruise trim deflection δ_e_cruise.
        d.  Verify  |δ_e_cruise| < δ_up_max  (Eq. 10 of the methodology:
            cruise deflection must be less than the takeoff up-limit).
4.  Accept the first bE/bh that satisfies both constraints.
5.  Derive SE/Sh = (cE/ch) × (bE/bh)  (rectangular-panel assumption).
6.  Recompute all derivatives with the final geometry and return.
"""
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import LLTResult
from pipeline.helpers import ScissorData
from propulsion.sizing import PropulsionResult
from sizing.wing import SizingResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ElevatorInputs:
    # User-defined sizing assumptions and limits.
    ih:    float = 0.0               # horizontal-tail incidence angle [rad]
    Z_T:   float = 0.0               # thrust-line vertical offset from CG [m]
    eta_h: float = 0.85          # dynamic-pressure ratio at the tail (Vh/V)²

    Cm_dist: float = 0.10            # max pitch-moment disturbance from cruise the
                                     # elevator must counteract at δ_up_max [-]

    max_deflection_up_deg:   float = 25.0   # elevator-up   deflection limit [deg]
    max_deflection_down_deg: float = 20.0   # elevator-down deflection limit [deg]

@dataclass
class ElevatorGeometry:
    """Derived geometry — output of the sizing loop, never a designer input."""
    bE_bh: float   # elevator span  / tail span  [-]
    cE_ch: float   # elevator chord / tail chord [-]
    SE_Sh: float   # elevator area  / tail area  [-]

@dataclass
class ElevatorResult:
    inputs:   ElevatorInputs
    geometry: ElevatorGeometry    # ← replaces the three ratio fields
    Vh:             float
    alpha_h:        float
    tau_e:          float
    CM_deltaE:      float
    CL_deltaE:      float
    CLh_deltaE:     float
    delta_e_cruise: float
    within_limits:  bool

# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationships
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """Control-surface effectiveness τ from chord ratio cf/c (empirical fit)."""
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


def cE_ch_from_tau(tau_e: float) -> float:
    """
    Elevator chord ratio cE/ch from effectiveness τ_e (polynomial inversion).
    Valid range: cE/ch ∈ [0.20, 0.40].
    """
    coeffs = [-6.624, 12.07, -8.292, 3.295, 0.004942 - tau_e]
    roots  = np.roots(coeffs)
    roots  = roots[np.isclose(roots.imag, 0)].real
    valid  = roots[(roots >= 0.20) & (roots <= 0.40)]
    if len(valid) == 0:
        raise ValueError(f"No valid elevator chord ratio for tau_e = {tau_e:.4f}")
    return float(valid[0])


# ---------------------------------------------------------------------------
# Main sizing routine
# ---------------------------------------------------------------------------

def run(
    sizing:       SizingResult,
    scissor:      ScissorData,
    llt:          LLTResult,
    propulsion:   PropulsionResult,
    polar:        AirfoilPolar,
    inputs:       ElevatorInputs | None = None,
) -> ElevatorResult:
    """
    Size the elevator by finding the minimum bE/bh (starting from 0.03) such
    that:
      (a) cE/ch ∈ [0.20, 0.40], and
      (b) the cruise trim deflection δ_e_cruise < δ_up_max  (Eq. 10).

    Parameters
    ----------
    sizing      : converged wing/tail sizing result
    scissor     : scissor-plot stability data
    llt         : LLT result carrying the cruise angle of attack
    propulsion  : propulsion result for thrust pitching moment
    polar       : airfoil polar for zero-lift angle
    inputs      : ElevatorInputs overrides; defaults used when None

    Returns
    -------
    ElevatorResult containing the converged elevator geometry and
    associated control derivatives.
    """
    if inputs is None:
        inputs = ElevatorInputs()
    i = inputs
    s = sizing

    # ------------------------------------------------------------------
    # Aero state from the converged design
    # ------------------------------------------------------------------
    CLalpha  = scissor.CL_alpha_w
    CLalphah = scissor.CL_alpha_h
    Cmalpha  = -scissor.CL_alpha_A_h * scissor.SM
    Cm0      = scissor.Cm_ac
    AR       = s.inputs.AR
    alpha    = llt.alpha_root
    CL0      = -CLalpha * polar.alpha_L0
    CLcr     = s.CL

    # Tail geometry coefficients
    Vh   = s.Sh * s.lh / (s.Sw * s.c)
    Sh_S = s.Sh / s.Sw

    # ------------------------------------------------------------------
    # Downwash → tail angle of attack at cruise
    # Both epsilon and ih are used here; alphah is fixed before the loop
    # since it depends only on wing geometry, not on elevator sizing.
    # ------------------------------------------------------------------
    dedalpha = (2 * CLalpha) / (np.pi * AR)
    epsilon0 = (2 * CL0)    / (np.pi * AR)
    epsilon  = epsilon0 + dedalpha * alpha
    alphah   = alpha + i.ih - epsilon

    # ------------------------------------------------------------------
    # Thrust pitching moment (geometry-independent)
    # ------------------------------------------------------------------
    T         = propulsion.thrust_cruise_per_prop * propulsion.inputs.n_props
    q         = s.q_cruise
    Cm_thrust = (T * i.Z_T) / (q * s.Sw * s.c)

    delta_e_max = np.radians(i.max_deflection_up_deg)

    # ------------------------------------------------------------------
    # Step 1 — solve the pitch-moment-disturbance requirement for tau_e
    # and cE/ch.  This is independent of bE/bh so it is done once before
    # the loop.
    #
    # At δ_e = δ_up_max the elevator-induced pitching moment must counteract
    # the maximum cruise pitch disturbance Cm_dist.  Written through the tail
    # volume (so it stays independent of the elevator span ratio):
    #
    #   |ΔCm| = CLalphah * tau_e * delta_e_max * eta_h * Vh = Cm_dist
    #   → tau_e = Cm_dist / (CLalphah * eta_h * Vh * delta_e_max)
    # ------------------------------------------------------------------
    tau_e_req = i.Cm_dist / (CLalphah * i.eta_h * Vh * delta_e_max)

    try:
        cE_ch_val = cE_ch_from_tau(tau_e_req)
    except ValueError:
        raise ValueError(
            f"Elevator sizing failed: the required effectiveness "
            f"tau_e = {tau_e_req:.4f} cannot be achieved within "
            f"cE/ch ∈ [0.20, 0.40].  Check tail volume, the pitch "
            f"disturbance Cm_dist, or the deflection limit."
        )

    # ------------------------------------------------------------------
    # Step 2 — iterate bE/bh to satisfy the cruise-deflection check
    # (Eq. 10: delta_e_cruise < delta_e_max)
    #
    # tau_e and cE/ch are the same for every iteration — they are set
    # by the tail-CL constraint above.  What changes with bE/bh is the
    # denominator of delta_e_cruise via CM_deltaE and CL_deltaE, so
    # bE/bh directly controls how much pitch authority the elevator has.
    # A larger span gives more authority, pulling delta_e_cruise down
    # until it satisfies Eq. 10.
    # ------------------------------------------------------------------
    found        = False
    chosen_bE_bh = None
    delta_e_sol  = None

    for bE_bh_val in np.arange(0.03, 0.121, 0.01):

        # Control derivatives at this candidate span ratio (Eq. 11, 12, 13)
        CM_dE_trial = -CLalphah * i.eta_h * Vh * bE_bh_val * tau_e_req
        CL_dE_trial =  CLalphah * i.eta_h * Vh * Sh_S * bE_bh_val * tau_e_req

        # Cruise trim deflection
        delta_e_cruise_trial = (
            (Cm_thrust + Cm0) * CLalpha
            + (CLcr - CL0) * Cmalpha
        ) / (CLalpha * CM_dE_trial - Cmalpha * CL_dE_trial)

        # Eq. 10: cruise deflection must be strictly inside the up-limit
        if abs(delta_e_cruise_trial) < delta_e_max:
            found        = True
            chosen_bE_bh = bE_bh_val
            delta_e_sol  = delta_e_cruise_trial
            break

    if not found:
        raise ValueError(
            "Elevator sizing failed: no bE/bh ∈ [0.03, 0.12] brings the "
            "cruise trim deflection below the up-limit of "
            f"{i.max_deflection_up_deg:.1f}°.\n"
            "Consider increasing tail volume, adjusting incidence, or "
            "relaxing the deflection limit."
        )

    # ------------------------------------------------------------------
    # Store converged elevator geometry
    # ------------------------------------------------------------------
    geometry = ElevatorGeometry(
        bE_bh = chosen_bE_bh,
        cE_ch = cE_ch_val,
        SE_Sh = cE_ch_val * chosen_bE_bh,
    )
    # ------------------------------------------------------------------
    # Final derivatives with converged geometry (Eq. 11, 12, 13)
    # ------------------------------------------------------------------
    CM_deltaE  = -CLalphah * i.eta_h * Vh * chosen_bE_bh * tau_e_req
    CL_deltaE  =  CLalphah * i.eta_h * Vh * Sh_S * chosen_bE_bh * tau_e_req
    CLh_deltaE =  CLalphah * tau_e_req

    within_limits = (
        -np.radians(i.max_deflection_down_deg)
        <= delta_e_sol
        <= np.radians(i.max_deflection_up_deg)
    )

    return ElevatorResult(
        inputs         = inputs,
        geometry       = geometry,
        Vh             = Vh,
        alpha_h        = alphah,
        tau_e          = tau_e_req,
        CM_deltaE      = CM_deltaE,
        CL_deltaE      = CL_deltaE,
        CLh_deltaE     = CLh_deltaE,
        delta_e_cruise = delta_e_sol,
        within_limits  = within_limits,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summary(r: ElevatorResult) -> None:
    i = r.inputs
    print(f"  Converged elevator geometry")
    g = r.geometry
    print(f"    bE/bh : {g.bE_bh:.2f}")
    print(f"    cE/ch : {g.cE_ch:.4f}")
    print(f"    SE/Sh : {g.SE_Sh:.4f}")
    print(f"  Effectiveness τ_e           : {r.tau_e:.3f}")
    print(f"  Tail AoA at cruise          : {np.degrees(r.alpha_h):.2f}  °")
    print(f"  CM_δe                       : {r.CM_deltaE:.4f}  1/rad")
    print(f"  CL_δe                       : {r.CL_deltaE:.4f}  1/rad")
    print(f"  CLh_δe                      : {r.CLh_deltaE:.4f}  1/rad")
    print(f"  Cruise trim deflection      : {np.degrees(r.delta_e_cruise):+.2f}  °")
    if r.within_limits:
        print(f"  ✓ Within [-{i.max_deflection_down_deg:.0f}°, "
              f"+{i.max_deflection_up_deg:.0f}°] deflection limits")
    else:
        print(f"  ✗ Cruise deflection EXCEEDS the "
              f"[-{i.max_deflection_down_deg:.0f}°, "
              f"+{i.max_deflection_up_deg:.0f}°] limits")