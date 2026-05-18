
import numpy as np

from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import FlightCondition, LLTResult, WingGeometry, solve_llt


def lift_slope_from_llt(llt: LLTResult) -> float:
    """3D lift-curve slope dCL/dα [1/rad] from a single existing LLT solve.

    CL is exactly linear in α_root in classical LLT
    (CL = CL_α · (α_root − α_L0_eff)), so a single point and the section's
    zero-lift angle pin down the slope without another solve.

    Assumes untwisted planform with uniform section α_L0 (the default
    `WingGeometry`/`AirfoilPolar` case in this codebase).
    """
    return llt.CL / (llt.alpha_root - llt.polar.alpha_L0)


def lift_slope_A_minus_h(
    CL_alpha_w: float,
    b: float,
    b_f: float,
    S: float,
    S_net: float,
) -> float:
    """Aircraft-minus-tail lift slope CL_α,A-h with wing+fuselage interference.

        CL_α,A-h = CL_α,w · (1 + 2.15·b_f/b) · S_net/S  +  (π/2) · b_f²/S
    """
    return CL_alpha_w * (1.0 + 2.15 * b_f / b) * S_net / S + 0.5 * np.pi * b_f**2 / S


def downwash_gradient(
    CL_alpha_w: float,
    A_wing: float,
    r: float,
    m_tv: float = 0.0,
    sweep_qc: float = 0.0,
) -> float:
    """Slingerland wing downwash gradient dε/dα [-] at the tail.

    `r = 2·l_h / b_wing`, `m_tv = 2·h_tail_above_zero_lift_line / b_wing`,
    `sweep_qc` is wing quarter-chord sweep in radians. With Λ=0 the K_ε ratio
    collapses to 1; with m_tv=0 the second bracket simplifies to its peak value.
    """
    K_eps_0 = 0.1124 / r**2 + 0.1024 / r + 2.0
    K_eps_L = (0.1124 + 0.1265 * sweep_qc + 0.1766 * sweep_qc**2) / r**2 + 0.1024 / r + 2.0

    term1 = (r / (r**2 + m_tv**2)) * 0.4876 / np.sqrt(r**2 + 0.6319 + m_tv**2)
    term2 = (1.0 + (r**2 / (r**2 + 0.7915 + 5.0734 * m_tv**2)) ** 0.3113) * (
        1.0 - np.sqrt(m_tv**2 / (1.0 + m_tv**2))
    )
    return (K_eps_L / K_eps_0) * (term1 + term2) * CL_alpha_w / (np.pi * A_wing)


def controllability_line_ShS(
    x_cg: np.ndarray,
    *,
    x_ac: float,
    c: float,
    l_h: float,
    CL_h: float,
    CL_A_h: float,
    Cm_ac: float,
    Vh_V: float,
) -> np.ndarray:
    """Required S_h/S as a function of x_cg for trim (controllability).

    Inverts the trim moment equation
        x̄_cg = x̄_ac − Cm_ac/CL_A-h + (CL_h/CL_A-h)(S_h/S)(l_h/c̄)(V_h/V)²
    for S_h/S. With a typical conventional tail at trim CL_h is negative
    (downforce), so the line slopes opposite to the stability line — together
    they form the classical scissor and bound the allowable CG range.

    `CL_h`, `CL_A_h`, `Cm_ac` are evaluated at the controllability condition
    (cruise as default; switch to landing/max-CL values for the conservative
    case).
    """
    factor = (CL_h / CL_A_h) * (l_h / c) * Vh_V**2
    return ((x_cg - x_ac) / c + Cm_ac / CL_A_h) / factor


def stability_line_ShS(
    x_cg: np.ndarray,
    *,
    x_ac: float,
    c: float,
    l_h: float,
    CL_alpha_h: float,
    CL_alpha_A_h: float,
    dep_da: float,
    Vh_V: float,
    SM: float,
) -> np.ndarray:
    """Required S_h/S as a function of x_cg for static longitudinal stability.

    Inverts the trim/stability equation
        x̄_cg = x̄_ac + (CL_α,h/CL_α,A-h)(1 − dε/dα)(S_h/S)(l_h/c̄)(V_h/V)² − SM
    for S_h/S. `x_cg`, `x_ac` and `c` must share a datum (LEMAC, in metres).
    """
    factor = (CL_alpha_h / CL_alpha_A_h) * (1.0 - dep_da) * (l_h / c) * Vh_V**2
    return ((x_cg - x_ac) / c + SM) / factor


def calculate_CM_wing(
    polar: AirfoilPolar,
    wing_geometry: WingGeometry,
    alpha_cruise: float,
) -> float:
    """Wing pitching moment about the aerodynamic center (3D finite-wing correction)."""
    Cm_2d = float(np.interp(alpha_cruise, polar.alpha, polar.Cm))
    return Cm_2d * wing_geometry.AR / (2 + wing_geometry.AR)


def calculate_tail_loading(
    wing_polar: AirfoilPolar,
    wing_geometry: WingGeometry,
    tail_polar: AirfoilPolar,
    tail_geometry: WingGeometry,
    flight: FlightCondition,
    *,
    alpha_cruise: float,
    CL_wing_cruise: float,
    l_tail: float,
    x_cg: float,
) -> dict:
    """Trim tail CL (no downwash) plus induced-drag properties from LLT on the tail planform.

    Moment balance about the CG, with l_tail measured from CG to tail AC (+aft)
    and x_ac,wing = 0.25 · c_mean (thin-airfoil approximation):

        CM_ac,wing + CL_w · (x_cg − x_ac,wing)/c  −  CL_t · (S_t/S_w)(l_t/c)  =  0
    """
    c = wing_geometry.c_mean
    x_ac_wing = 0.25 * c

    CM_ac_wing = calculate_CM_wing(wing_polar, wing_geometry, alpha_cruise)

    V_H = (tail_geometry.S / wing_geometry.S) * (l_tail / c)
    CL_tail = (CM_ac_wing + CL_wing_cruise * (x_cg - x_ac_wing) / c) / V_H

    tail_flight = FlightCondition(
        V_inf=flight.V_inf,
        rho=flight.rho,
        CL_target=CL_tail,
    )
    tail_result = solve_llt(tail_geometry, tail_polar, tail_flight)

    return {
        "CL_tail": CL_tail,
        "CD_i_tail": tail_result.CD_i,
        "e_tail": tail_result.e,
        "AR_tail": tail_geometry.AR,
        "llt_tail": tail_result,
    }

def calculate_lh(
    x_cg: np.ndarray,
    *,
    x_ac: float,
    c: float,
    Sh_S: float,
    CL_h: float,
    CL_A_h: float,
    Cm_ac: float,
    Vh_V: float,
    CL_alpha_h: float,
    CL_alpha_A_h: float,
    dep_da: float,
    SM: float,
) -> np.ndarray:

    factor_stab = (CL_alpha_h / CL_alpha_A_h) * (1.0 - dep_da) * Sh_S * Vh_V**2
    factor_cont = (CL_h / CL_A_h) * Sh_S * Vh_V**2

    lh_c_stab = factor_stab / ((x_cg - x_ac / c) + (Cm_ac / CL_A_h))
    lh_c_cont = factor_cont / ((x_cg - x_ac / c) + SM)

    return max(lh_c_stab, lh_c_cont)