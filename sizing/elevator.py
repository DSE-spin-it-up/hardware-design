# Positive elevator deflection (elevator down) produces nose down pitch so Cldeltae is -ve.
"""Elevator sizing: required cruise-trim deflection from the tail/elevator geometry.

Given the converged wing+tail state (lift slopes and Cm from the stability
scissor, the trim tail-CL from the cruise tail loading, and the cruise thrust),
this computes the elevator effectiveness τ_e needed to trim, the implied
elevator/tail chord ratio, the elevator control derivatives, and the actual
elevator deflection required at cruise. The cruise deflection is then checked
against the up/down deflection limits.
"""
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import LLTResult
from pipeline.helpers import ScissorData
from propulsion.sizing import PropulsionResult
from sizing.wing import SizingResult


@dataclass
class ElevatorInputs:
    SE_Sh: float = 0.25       # elevator area / horizontal-tail area [-] (0.15–0.40)
    bE_bh: float = 0.08       # elevator span / horizontal-tail span [-] (0.03–0.12)
    ih: float = 0.0           # horizontal-tail incidence angle [rad]
    Z_T: float = 0.0          # thrust-line vertical offset from CG [m] (2·h_wing_prop + h_tail_prop)/3
    eta_h: float = 0.85 ** 2  # dynamic-pressure ratio at the tail, (V_h/V)² [-]
    max_deflection_up_deg: float = 25.0    # elevator-up deflection limit [deg]
    max_deflection_down_deg: float = 20.0  # elevator-down deflection limit [deg]


@dataclass
class ElevatorResult:
    inputs: ElevatorInputs
    Vh: float            # horizontal-tail volume coefficient [-]
    alpha_h: float       # horizontal-tail angle of attack at cruise [rad]
    tau_e: float         # elevator effectiveness needed to trim [-]
    cE_ch: float         # elevator chord / tail chord [-]
    CM_deltaE: float     # pitching-moment control derivative [1/rad]
    CL_deltaE: float     # aircraft lift control derivative [1/rad]
    CLh_deltaE: float    # tail-lift control derivative [1/rad]
    delta_e_cruise: float  # required elevator deflection at cruise [rad]
    within_limits: bool    # cruise deflection inside up/down limits


def cE_ch(tau_e: float) -> float:
    """Elevator/tail chord ratio from effectiveness τ_e (inverts the empirical fit)."""
    coeffs = [-6.624, 12.07, -8.292, 3.295, 0.004942 - tau_e]
    roots = np.roots(coeffs)
    roots = roots[np.isclose(roots.imag, 0)].real
    valid = roots[(roots >= 0.2) & (roots <= 0.4)]
    if len(valid) == 0:
        raise ValueError(f"No valid elevator chord ratio for tau_e={tau_e:.4f}")
    return float(valid[0])


def run(
    sizing: SizingResult,
    scissor: ScissorData,
    llt: LLTResult,
    tail_loading: dict,
    propulsion: PropulsionResult,
    polar: AirfoilPolar,
    inputs: ElevatorInputs | None = None,
) -> ElevatorResult:
    if inputs is None:
        inputs = ElevatorInputs()
    i = inputs
    s = sizing

    # --- Pull the aero state from the converged design ---
    CLalpha = scissor.CL_alpha_w        # wing lift-curve slope [1/rad]
    CLalphah = scissor.CL_alpha_h       # horizontal-tail lift-curve slope [1/rad]
    Cmalpha = -scissor.CL_alpha_A_h * scissor.SM  # aircraft pitch-stiffness slope [1/rad]
    Cm0 = scissor.Cm_ac                 # zero-lift (wing AC) pitching moment [-]
    AR = s.inputs.AR
    alpha = llt.alpha_root              # cruise angle of attack [rad]
    CL0 = -CLalpha * polar.alpha_L0     # wing CL at α = 0 [-]
    CLcr = s.CL                         # cruise CL [-]
    CLh = tail_loading["CL_tail"]       # trim tail CL [-]

    # Horizontal-tail volume coefficient and area ratio from the geometry.
    Vh = s.Sh * s.lh / (s.Sw * s.c)
    Sh_S = s.Sh / s.Sw

    # --- Downwash → tail angle of attack at cruise ---
    dedalpha = (2 * CLalpha) / (np.pi * AR)
    epsilon0 = (2 * CL0) / (np.pi * AR)
    epsilon = epsilon0 + dedalpha * alpha
    alphah = alpha + i.ih - epsilon

    # --- Elevator effectiveness needed to reach the trim tail CL at the up limit ---
    delta_up = np.radians(inputs.max_deflection_up_deg)
    tau_e = (CLh - CLalphah * alphah) / (CLalphah * delta_up)
    cE_ch_ratio = cE_ch(tau_e)

    # --- Control derivatives ---
    CM_deltaE = -CLalphah * i.eta_h * Vh * i.bE_bh * tau_e
    CL_deltaE = CLalphah * i.eta_h * Vh * Sh_S * i.bE_bh * tau_e
    CLh_deltaE = CLalphah * tau_e

    # --- Required elevator deflection at cruise (thrust pitching moment included) ---
    T = propulsion.thrust_cruise_per_prop * propulsion.inputs.n_props
    q = s.q_cruise
    delta_e_cruise = (
        ((T * i.Z_T) / (q * s.Sw * s.c) + Cm0) * CLalpha
        + (CLcr - CL0) * Cmalpha
    ) / (CLalpha * CM_deltaE - Cmalpha * CL_deltaE)

    within_limits = (
        -np.radians(i.max_deflection_down_deg)
        <= delta_e_cruise
        <= np.radians(i.max_deflection_up_deg)
    )

    return ElevatorResult(
        inputs=inputs,
        Vh=Vh,
        alpha_h=alphah,
        tau_e=tau_e,
        cE_ch=cE_ch_ratio,
        CM_deltaE=CM_deltaE,
        CL_deltaE=CL_deltaE,
        CLh_deltaE=CLh_deltaE,
        delta_e_cruise=delta_e_cruise,
        within_limits=within_limits,
    )


def summary(r: ElevatorResult) -> None:
    print(f"  Elevator/tail chord ratio : {r.cE_ch:.3f}")
    print(f"  Effectiveness τ_e         : {r.tau_e:.3f}")
    print(f"  Tail AoA at cruise        : {np.degrees(r.alpha_h):.2f}  °")
    print(f"  CM_δe                     : {r.CM_deltaE:.4f}  1/rad")
    print(f"  CL_δe                     : {r.CL_deltaE:.4f}  1/rad")
    print(f"  Cruise trim deflection    : {np.degrees(r.delta_e_cruise):+.2f}  °")
    if r.within_limits:
        print(f"  ✓ Within ±[{r.inputs.max_deflection_down_deg:.0f},"
              f"{r.inputs.max_deflection_up_deg:.0f}]° deflection limits")
    else:
        print(f"  ✗ Cruise deflection EXCEEDS the "
              f"[-{r.inputs.max_deflection_down_deg:.0f},"
              f"+{r.inputs.max_deflection_up_deg:.0f}]° limits")
