"""Aileron sizing: pick inboard span so the ailerons can counteract a roll disturbance.

Mirrors the elevator approach: at the maximum deflection δ_a_max the aileron
roll-control moment must counteract the maximum roll-moment disturbance
Cl_dist (an input).  The loop grows the aileron inboard from the outboard
edge until |Cl_δa · δ_a_max| ≥ Cl_dist.  The achievable steady roll rate is
still reported (it uses the roll-damping derivative Cl_p) but no longer gates
the sizing.

A second sizing criterion mirrors the elevator payload check: a horizontal
payload tension (magnitude m_payload × g) acting at the PVC tube bottom
produces a roll moment whose arm is the vertical distance from that attach
point to the overall CG.  Whichever criterion — disturbance or payload —
demands the larger inboard span drives the result.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from aerodynamics.airfoil_polar import AirfoilPolar
from sizing.wing import SizingResult


@dataclass
class AileronInputs:
    cl_alpha: float = 2 * np.pi      # section lift slope [1/rad]
    cd0_section: float = 0.04        # fallback section profile drag if no polar supplied [-]
    roll_moment_dist: float = 0.05   # max roll-moment disturbance the ailerons
                                     # must counteract at δ_a_max [-]
    max_da_deg: float = 10.0         # max aileron deflection [deg]
    max_y_frac: float = 0.8          # outboard aileron edge, y/(b/2) [-]
    c_aileron_to_c_wing: float = 0.3 # aileron chord / wing chord [-]
    step: float = 0.01               # search step on y_frac [-]


@dataclass
class AileronResult:
    inputs: AileronInputs
    start_y_frac: float    # inboard aileron edge, y/(b/2)
    aileron_span: float    # per-side span [m]
    tau: float             # control-surface effectiveness [-]
    cd0_section: float     # section profile drag actually used in Cl_p [-]
    cl_da: float           # roll control derivative [1/rad]
    cl_p: float            # roll damping derivative [1/rad]
    roll_moment: float     # roll-control moment at δ_a_max, |Cl_δa·δ_a_max| [-]
    roll_rate: float       # achieved steady roll rate [rad/s]
    converged: bool        # True if disturbance counteracted before hitting root
    roll_moment_payload: float   # roll moment from payload horizontal tension [-]
    driving_constraint: str      # "disturbance" or "payload"


def chord_at_y_frac(c_root: float, lam: float):
    """Linear-taper chord as a function of y/(b/2)."""
    def f(y_frac: float) -> float:
        return c_root * (1 - (1 - lam) * y_frac)
    return f


def tau_from_ratio(c_aileron_to_c_wing: float) -> float:
    """Aileron effectiveness τ from chord ratio (piecewise empirical fit)."""
    if c_aileron_to_c_wing > 0.7:
        raise ValueError(
            f"c_aileron/c_wing = {c_aileron_to_c_wing} is too high (>0.7)."
        )
    if c_aileron_to_c_wing < 0.2:
        return 2 * c_aileron_to_c_wing
    return 0.4 + (0.4 / 0.5) * c_aileron_to_c_wing


def compute_cl_da(
    start_y_frac: float,
    end_y_frac: float,
    c_at_y_frac,
    b: float,
    S_ref: float,
    cl_alpha: float,
    tau: float,
) -> float:
    """Cl_δa = (2 cl_α τ) / (S b) * ∫ c(y) y dy, with y in physical units."""
    # y_phys = y_frac * b/2, dy_phys = (b/2) dy_frac → factor (b/2)^2
    integrand = lambda y: c_at_y_frac(y) * y * (b / 2) ** 2
    c_integral, _ = quad(integrand, start_y_frac, end_y_frac)
    return (2 * cl_alpha * tau) / (S_ref * b) * c_integral


def compute_cl_p(
    cl_alpha: float,
    cd0_section: float,
    S_ref: float,
    b: float,
    c_at_y_frac,
) -> float:
    """Cl_p = -(4 (cl_α + cd0)) / (S b²) * ∫ c(y) y² dy."""
    integrand = lambda y: c_at_y_frac(y) * (y ** 2) * (b / 2) ** 3
    c_integral, _ = quad(integrand, 0.0, 1.0)
    return -(4 * (cl_alpha + cd0_section) / (S_ref * b ** 2)) * c_integral


def run(
    sizing: SizingResult,
    inputs: AileronInputs | None = None,
    polar: AirfoilPolar | None = None,
    y_cg: dict[str, float] | None = None,
) -> AileronResult:
    """Size the ailerons.

    Parameters
    ----------
    sizing  : converged wing sizing result
    inputs  : AileronInputs overrides; defaults used when None
    polar   : wing airfoil polar for section Cd_p; fallback used when None
    y_cg    : vertical CG dict from compute_y_cg (keys: 'overall',
              'pvc_tube_bottom').  When supplied, a payload roll-moment
              criterion is added alongside the disturbance criterion,
              exactly mirroring the elevator payload check.  When None
              (e.g. standalone ``__main__`` runs) only the disturbance
              criterion is active.
    """
    if inputs is None:
        inputs = AileronInputs()
    i = inputs
    s = sizing

    c_at_y_frac = chord_at_y_frac(s.c_root, s.inputs.lam)
    tau = tau_from_ratio(i.c_aileron_to_c_wing)
    max_da = np.deg2rad(i.max_da_deg)

    # Section profile drag at the operating Cl — pulled from the XFOIL polar
    # when available, otherwise the (rougher) fallback from inputs.
    if polar is not None:
        cd0_section = float(polar.Cd_p(s.Cl_airfoil))
    else:
        cd0_section = i.cd0_section

    cl_p = compute_cl_p(i.cl_alpha, cd0_section, s.Sw, s.inputs.b, c_at_y_frac)

    # ------------------------------------------------------------------
    # Payload roll-moment criterion
    #
    # A horizontal payload tension (m_payload × g) acts at the PVC tube
    # bottom attach point.  The moment arm is the vertical distance from
    # that point to the overall CG — identical to the elevator payload
    # arm — because the force is horizontal and the arm is vertical.
    #
    #   Cl_payload = (F_payload × moment_arm) / (q × Sw × b)
    #
    # When y_cg is not supplied (standalone runs) the payload criterion
    # is suppressed and only the disturbance requirement drives sizing.
    # ------------------------------------------------------------------
    if y_cg is not None:
        moment_arm = abs(y_cg["overall"] - y_cg["pvc_tube_bottom"])
        F_payload = s.inputs.m_payload * 9.81
        roll_moment_payload = (F_payload * moment_arm) / (s.q_cruise * s.Sw * s.inputs.b)
    else:
        roll_moment_payload = 0.0

    roll_moment_required = max(i.roll_moment_dist, roll_moment_payload)
    driving_constraint = "payload" if roll_moment_payload > i.roll_moment_dist else "disturbance"

    # ------------------------------------------------------------------
    # Inboard-span search
    # ------------------------------------------------------------------
    start_y_frac = i.max_y_frac
    cl_da = 0.0
    roll_moment = 0.0
    converged = False
    while start_y_frac > 0.0:
        start_y_frac -= i.step
        cl_da = compute_cl_da(
            start_y_frac, i.max_y_frac, c_at_y_frac,
            s.inputs.b, s.Sw, i.cl_alpha, tau,
        )
        # Roll-control moment available at full deflection; size until it
        # counteracts whichever criterion (disturbance or payload) is larger.
        roll_moment = abs(cl_da * max_da)
        if roll_moment >= roll_moment_required:
            converged = True
            break

    # Steady roll rate at the converged geometry — reported, not a sizing gate.
    P = -(cl_da / cl_p) * max_da * (2 * s.inputs.V_cruise / s.inputs.b)
    aileron_span = (i.max_y_frac - start_y_frac) * (s.inputs.b / 2)
    return AileronResult(
        inputs=inputs,
        start_y_frac=start_y_frac,
        aileron_span=aileron_span,
        tau=tau,
        cd0_section=cd0_section,
        cl_da=cl_da,
        cl_p=cl_p,
        roll_moment=roll_moment,
        roll_rate=P,
        converged=converged,
        roll_moment_payload=roll_moment_payload,
        driving_constraint=driving_constraint,
    )


def summary(r: AileronResult) -> None:
    req = max(r.inputs.roll_moment_dist, r.roll_moment_payload)
    if r.converged:
        print(f"  Roll-moment requirement met "
              f"({r.roll_moment:.4f} ≥ {req:.4f})")
    else:
        print(f"  ✗ Roll-moment requirement NOT met "
              f"({r.roll_moment:.4f} < {req:.4f})")
    print(f"  Active constraint           : {r.driving_constraint}")
    print(f"  Roll-moment (disturbance)   : {r.inputs.roll_moment_dist:.4f}")
    print(f"  Roll-moment (payload)       : {r.roll_moment_payload:.4f}")
    print(f"  Aileron inboard y/(b/2) : {r.start_y_frac:.3f}")
    print(f"  Aileron outboard y/(b/2): {r.inputs.max_y_frac:.3f}")
    print(f"  Aileron span (per side) : {r.aileron_span:.3f}  m")
    print(f"  τ (effectiveness)       : {r.tau:.3f}")
    print(f"  cd0 (section, used)     : {r.cd0_section:.5f}")
    print(f"  Cl_δa                   : {r.cl_da:.4f}  1/rad")
    print(f"  Cl_p                    : {r.cl_p:.4f}  1/rad")
    print(f"  Roll-control moment     : {r.roll_moment:.4f}  (at δ_a_max)")
    print(f"  Steady roll rate        : {np.degrees(r.roll_rate):.2f}  °/s")


if __name__ == "__main__":
    from sizing import wing
    summary(run(wing.run()))