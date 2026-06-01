# Positive rudder deflection (rudder to the left) yaws nose the left so Cldeltar is -ve.
"""Rudder sizing: rudder deflection needed to hold a crosswind gust.

Solves the steady yaw + side-force balance under a lateral gust for the
sideslip σ the airframe weathercocks to and the rudder deflection δ_R needed
to keep it trimmed, given the rudder geometry ratios (S_r/S_v, c_r/c_v,
b_r/b_v). The required deflection is then checked against the 30° rudder
limit. Vertical-tail derivatives (Cn_β, Cy_β, Cy_δr, Cn_δr) are built from the
converged tail geometry and lift slope.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from pipeline.helpers import ScissorData
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult


@dataclass
class RudderInputs:
    SR_SV: float = 0.25   # rudder area / vertical-tail area [-] (0.15–0.35)
    cR_CV: float = 0.28   # rudder chord / vertical-tail chord [-] (0.15–0.40)
    bR_bV: float = 0.85   # rudder span / vertical-tail span [-] (0.70–1.00)
    max_deflection_deg: float = 30.0  # rudder deflection limit [deg]
    Vgust: float = 5.0    # lateral gust velocity [m/s]
    CDY: float = 0.8      # side drag coefficient for the gust force [-] (0.5–0.8)
    Kf1: float = 0.85     # fuselage correction on Cn_β [-] (0.65–0.85)
    Kf2: float = 1.0      # fuselage correction on Cy_β [-] (0.75–1.0)
    eta_v: float = 0.85 ** 2  # dynamic-pressure ratio at the vertical tail, (V_v/V)² [-]
    dc: float = 0.0       # gust-force line angle in the yaw balance [rad]
    Ss: float | None = None   # body side area [m²]; derived from the fuselage when None


@dataclass
class RudderResult:
    inputs: RudderInputs
    CLalphav: float    # vertical-tail lift-curve slope [1/rad]
    tau_r: float       # rudder effectiveness [-]
    Cnb: float         # yaw stiffness derivative [1/rad]
    Cyb: float         # side-force/sideslip derivative [1/rad]
    Cndr: float        # yaw control derivative [1/rad]
    Cydr: float        # side-force control derivative [1/rad]
    beta_gust: float   # gust sideslip angle [rad]
    sigma: float       # weathercock sideslip [rad]
    delta_R: float     # required rudder deflection [rad]
    within_limits: bool  # required deflection inside the rudder limit


def tau_from_chord_ratio(cf_c: float) -> float:
    """Control-surface effectiveness τ from chord ratio c_f/c (empirical fit)."""
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], cf_c))


def cR_cv(tau_r: float) -> float:
    """Rudder/vertical-tail chord ratio from effectiveness τ_r (inverts the fit)."""
    coeffs = [-6.624, 12.07, -8.292, 3.295, 0.004942 - tau_r]
    roots = np.roots(coeffs)
    roots = roots[np.isclose(roots.imag, 0)].real
    valid = roots[(roots >= 0.15) & (roots <= 0.4)]
    if len(valid) == 0:
        raise ValueError(f"No valid rudder chord ratio for tau_r={tau_r:.4f}")
    return float(valid[0])


def _equations(x, beta, rho, VT, Vgust, S, b,
               Cn0, Cnb, Cndr, Cy0, Cyb, Cydr, CDY, Fw, dc):
    sigma, delta_R = x
    eq1 = (
        0.5 * rho * VT ** 2 * S * b *
        (Cn0 + Cnb * (beta - sigma) + Cndr * delta_R)
        + Fw * np.cos(dc) * np.cos(sigma)
    )
    eq2 = (
        0.5 * rho * Vgust ** 2 * S * CDY
        - 0.5 * rho * VT ** 2 * S *
        (Cy0 + Cyb * (beta - sigma) + Cydr * delta_R)
    )
    return [eq1, eq2]


def run(
    sizing: SizingResult,
    scissor: ScissorData,
    fus: FuselageResult,
    v_stall: float,
    inputs: RudderInputs | None = None,
) -> RudderResult:
    if inputs is None:
        inputs = RudderInputs()
    i = inputs
    s = sizing

    rho = s.rho
    S = s.Sw
    b = s.inputs.b
    Sv = s.Sv
    bv = s.bv
    lv = s.lh  # vertical-tail arm = tail length

    # Body side area for the gust side force.
    Ss = i.Ss if i.Ss is not None else fus.length * fus.height

    # Vertical-tail lift-curve slope from its (effective) aspect ratio.
    ARv = bv ** 2 / Sv
    CLalphav = 2 * np.pi * ARv / (ARv + 2)

    # Tail-volume coefficient and rudder geometry from the ratios.
    Vv = Sv * lv / (S * b)
    taur = tau_from_chord_ratio(i.cR_CV)
    br = i.bR_bV * bv
    Sr = i.SR_SV * Sv

    # Symmetric airframe → zero baseline yaw / side force.
    Cn0 = 0.0
    Cy0 = 0.0

    # Vertical-tail control / stability derivatives.
    Cyb = -i.Kf2 * CLalphav * i.eta_v * Sv / S
    Cnb = i.Kf1 * CLalphav * i.eta_v * Sv * lv / (b * S)
    Cydr = CLalphav * i.eta_v * taur * br / bv * Sr / Sv
    Cndr = -CLalphav * Vv * i.eta_v * taur * br / bv

    # Gust kinematics.
    VT = np.sqrt(v_stall ** 2 + i.Vgust ** 2)
    beta = np.arctan(i.Vgust / v_stall)
    Fw = 0.5 * rho * i.Vgust ** 2 * Ss * i.CDY

    sigma, delta_R = fsolve(
        _equations,
        [0.0, 0.0],
        args=(beta, rho, VT, i.Vgust, S, b,
              Cn0, Cnb, Cndr, Cy0, Cyb, Cydr, i.CDY, Fw, i.dc),
    )

    within_limits = abs(delta_R) <= np.radians(i.max_deflection_deg)

    return RudderResult(
        inputs=inputs,
        CLalphav=CLalphav,
        tau_r=taur,
        Cnb=Cnb,
        Cyb=Cyb,
        Cndr=Cndr,
        Cydr=Cydr,
        beta_gust=beta,
        sigma=sigma,
        delta_R=delta_R,
        within_limits=within_limits,
    )


def summary(r: RudderResult) -> None:
    print(f"  Vertical-tail CL_α        : {r.CLalphav:.3f}  1/rad")
    print(f"  Effectiveness τ_r         : {r.tau_r:.3f}")
    print(f"  Cn_β                      : {r.Cnb:.4f}  1/rad")
    print(f"  Cy_β                      : {r.Cyb:.4f}  1/rad")
    print(f"  Cn_δr                     : {r.Cndr:.4f}  1/rad")
    print(f"  Cy_δr                     : {r.Cydr:.4f}  1/rad")
    print(f"  Gust sideslip β           : {np.degrees(r.beta_gust):.2f}  °")
    print(f"  Weathercock σ             : {np.degrees(r.sigma):.2f}  °")
    print(f"  Required rudder deflection: {np.degrees(r.delta_R):+.2f}  °")
    if r.within_limits:
        print(f"  ✓ Within ±{r.inputs.max_deflection_deg:.0f}° deflection limit")
    else:
        print(f"  ✗ Required deflection EXCEEDS the "
              f"±{r.inputs.max_deflection_deg:.0f}° limit")
