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
    cE_ch:                   float = 0.30   # elevator chord / tail chord [-] (designer choice)
    eta_h:                   float = 0.85   # dynamic-pressure ratio at the tail (Vh/V)²
    Cm_dist:                 float = 0.10   # pitch-moment disturbance the elevator must
                                            # counteract at delta_e_max [-]
    max_deflection_up_deg:   float = 25.0   # elevator-up   deflection limit [deg]
    max_deflection_down_deg: float = 20.0   # elevator-down deflection limit [deg]


@dataclass
class ElevatorGeometry:
    """Derived geometry — output of sizing, never a designer input."""
    bE_bh: float   # elevator span  / tail span  [-]
    cE_ch: float   # elevator chord / tail chord [-]
    SE_Sh: float   # elevator area  / tail area  [-]


@dataclass
class ElevatorResult:
    inputs:         ElevatorInputs
    geometry:       ElevatorGeometry
    ih:             float           # tail incidence angle from trim [rad]
    Vh:             float           # tail volume coefficient [-]
    alpha_h:        float           # tail angle of attack at cruise [rad]
    tau_e:          float           # elevator effectiveness [-]
    CM_deltaE:      float           # dCm/d(delta_e) [1/rad]
    CL_deltaE:      float           # dCL/d(delta_e) [1/rad]
    CLh_deltaE:     float           # dCLh/d(delta_e) [1/rad]


# ---------------------------------------------------------------------------
# Empirical τ ↔ chord-ratio relationships
# ---------------------------------------------------------------------------

def tau_from_chord_ratio(cf_c: float) -> float:
    """Control-surface effectiveness τ from chord ratio cf/c (empirical fit)."""
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


# ---------------------------------------------------------------------------
# Main sizing routine
# ---------------------------------------------------------------------------

def run(
    sizing:     SizingResult,
    scissor:    ScissorData,
    llt:        LLTResult,
    propulsion: PropulsionResult,
    polar:      AirfoilPolar,
    y_cg:       dict[str, float],
    inputs:     ElevatorInputs | None = None,
) -> ElevatorResult:
    """
    Size the elevator by:
      1. Solving for ih from moment equilibrium at delta_e = 0.
      2. Fixing cE/ch (from inputs), deriving tau_e, then solving for bE/bh
         from the disturbance requirement.

    Parameters
    ----------
    sizing      : converged wing/tail sizing result
    scissor     : scissor-plot stability data
    llt         : LLT result carrying the cruise angle of attack
    propulsion  : propulsion result for thrust pitching moment
    polar       : airfoil polar for zero-lift angle
    y_cg        : vertical CG dict from compute_y_cg (keys: 'overall',
                  'motors', 'motor_back')
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
    CL0      = -CLalpha * polar.alpha_L0
    dedalpha = (2 * CLalpha) / (np.pi * AR)
    epsilon0 = (2 * CL0)    / (np.pi * AR)
    epsilon  = epsilon0 + dedalpha * alpha

    # ------------------------------------------------------------------
    # Thrust pitching moments about CG
    # 2 front propellers at y_cg["motors"], 1 back at y_cg["motor_back"].
    # Z_T positive when motor is above CG (nose-up moment for puller config).
    # ------------------------------------------------------------------
    y_overall = y_cg["overall"]
    Z_T_front = y_cg["motors"]     - y_overall
    Z_T_back  = y_cg["motor_back"] - y_overall

    T_per_prop = propulsion.thrust_cruise_per_prop
    q          = s.q_cruise

    Cm_thrust = (
        2.0 * T_per_prop * Z_T_front
        + 1.0 * T_per_prop * Z_T_back
    ) / (q * s.Sw * s.c)

    # ------------------------------------------------------------------
    # Solve for ih from moment equilibrium at delta_e = 0
    #
    # Cm_total = Cm0 + Cmalpha*(alpha - alpha_L0)          [wing-body]
    #          + Cm_thrust                                  [propellers]
    #          + CLalphah * eta_h * Vh                      [tail]
    #            * (alpha - epsilon + ih - alpha_L0_h)
    #          = 0
    #
    # alpha_L0_h = 0 for a symmetric tail airfoil.
    # ------------------------------------------------------------------
    alpha_L0   = polar.alpha_L0
    alpha_L0_h = 0.0   # symmetric tail airfoil assumption

    Cm_wing_body = Cm0 + Cmalpha * (alpha - alpha_L0)
    Cm_tail_base = CLalphah * i.eta_h * Vh * (alpha - epsilon - alpha_L0_h)

    ih = -(Cm_wing_body + Cm_thrust + Cm_tail_base) / (CLalphah * i.eta_h * Vh)

    # Tail angle of attack at cruise with solved ih
    alpha_h = alpha - epsilon + ih - alpha_L0_h

    # ------------------------------------------------------------------
    # Elevator sizing
    #
    # 1. Fix cE/ch (designer input) → tau_e from empirical polynomial.
    # 2. Solve for bE/bh from disturbance requirement:
    #      Cm_dist = CLalphah * eta_h * Vh * tau_e * delta_e_max * bE_bh
    #      → bE_bh = Cm_dist / (CLalphah * eta_h * Vh * tau_e * delta_e_max)
    # ------------------------------------------------------------------
    delta_e_max = np.radians(i.max_deflection_up_deg)

    tau_e = tau_from_chord_ratio(i.cE_ch)

    bE_bh = i.Cm_dist / (CLalphah * i.eta_h * Vh * tau_e * delta_e_max)

    if bE_bh > 1.0:
        raise ValueError(
            f"Elevator sizing failed: required bE/bh = {bE_bh:.3f} > 1.0.\n"
            "The full tail span is insufficient to counteract Cm_dist.\n"
            "Consider increasing cE/ch, increasing tail volume, or reducing Cm_dist."
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
    CL_deltaE  =  CLalphah * i.eta_h * Vh * Sh_S * bE_bh * tau_e
    CLh_deltaE =  CLalphah * tau_e

    return ElevatorResult(
        inputs     = inputs,
        geometry   = geometry,
        ih         = ih,
        Vh         = Vh,
        alpha_h    = alpha_h,
        tau_e      = tau_e,
        CM_deltaE  = CM_deltaE,
        CL_deltaE  = CL_deltaE,
        CLh_deltaE = CLh_deltaE,
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def summary(r: ElevatorResult) -> None:
    i = r.inputs
    g = r.geometry
    print(f"  Tail incidence angle ih     : {np.degrees(r.ih):+.3f}  °")
    print(f"  Tail volume coefficient Vh  : {r.Vh:.4f}")
    print(f"  Tail AoA at cruise          : {np.degrees(r.alpha_h):.2f}  °")
    print(f"  Elevator geometry")
    print(f"    cE/ch : {g.cE_ch:.4f}  (designer input)")
    print(f"    tau_e : {r.tau_e:.4f}")
    print(f"    bE/bh : {g.bE_bh:.4f}  (from disturbance sizing)")
    print(f"    SE/Sh : {g.SE_Sh:.4f}")
    print(f"  CM_δe                       : {r.CM_deltaE:.4f}  1/rad")
    print(f"  CL_δe                       : {r.CL_deltaE:.4f}  1/rad")
    print(f"  CLh_δe                      : {r.CLh_deltaE:.4f}  1/rad")