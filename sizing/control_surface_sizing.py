"""Aileron sizing: pick inboard span so steady roll rate meets the requirement."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from aerodynamics.airfoil_polar import AirfoilPolar
from sizing.initial_sizing import SizingResult


@dataclass
class ControlSurfaceInputs:
    cl_alpha: float = 2 * np.pi      # section lift slope [1/rad]
    cd0_section: float = 0.04        # fallback section profile drag if no polar supplied [-]
    roll_req_deg: float = 60.0       # required steady roll rate [deg/s]
    max_da_deg: float = 10.0         # max aileron deflection [deg]
    max_y_frac: float = 0.8          # outboard aileron edge, y/(b/2) [-]
    c_aileron_to_c_wing: float = 0.3 # aileron chord / wing chord [-]
    step: float = 0.01               # search step on y_frac [-]


@dataclass
class ControlSurfaceResult:
    inputs: ControlSurfaceInputs
    start_y_frac: float    # inboard aileron edge, y/(b/2)
    aileron_span: float    # per-side span [m]
    tau: float             # control-surface effectiveness [-]
    cd0_section: float     # section profile drag actually used in Cl_p [-]
    cl_da: float           # roll control derivative [1/rad]
    cl_p: float            # roll damping derivative [1/rad]
    roll_rate: float       # achieved steady roll rate [rad/s]
    converged: bool        # True if requirement met before hitting root


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
    inputs: ControlSurfaceInputs | None = None,
    polar: AirfoilPolar | None = None,
) -> ControlSurfaceResult:
    if inputs is None:
        inputs = ControlSurfaceInputs()
    i = inputs
    s = sizing

    c_at_y_frac = chord_at_y_frac(s.c_root, s.inputs.lam)
    tau = tau_from_ratio(i.c_aileron_to_c_wing)
    roll_req = np.deg2rad(i.roll_req_deg)
    max_da = np.deg2rad(i.max_da_deg)

    # Section profile drag at the operating Cl — pulled from the XFOIL polar
    # when available, otherwise the (rougher) fallback from inputs.
    if polar is not None:
        cd0_section = float(polar.Cd_p(s.Cl_airfoil))
    else:
        cd0_section = i.cd0_section

    cl_p = compute_cl_p(i.cl_alpha, cd0_section, s.Sw, s.inputs.b, c_at_y_frac)

    start_y_frac = i.max_y_frac
    cl_da = 0.0
    P = 0.0
    converged = False
    while start_y_frac > 0.0:
        start_y_frac -= i.step
        cl_da = compute_cl_da(
            start_y_frac, i.max_y_frac, c_at_y_frac,
            s.inputs.b, s.Sw, i.cl_alpha, tau,
        )
        P = -(cl_da / cl_p) * max_da * (2 * s.inputs.V_cruise / s.inputs.b)
        if P >= roll_req:
            converged = True
            break

    aileron_span = (i.max_y_frac - start_y_frac) * (s.inputs.b / 2)
    return ControlSurfaceResult(
        inputs=inputs,
        start_y_frac=start_y_frac,
        aileron_span=aileron_span,
        tau=tau,
        cd0_section=cd0_section,
        cl_da=cl_da,
        cl_p=cl_p,
        roll_rate=P,
        converged=converged,
    )


def summary(r: ControlSurfaceResult) -> None:
    if r.converged:
        print(f"  Roll-rate requirement met "
              f"({np.degrees(r.roll_rate):.2f} ≥ {r.inputs.roll_req_deg:.1f} °/s)")
    else:
        print(f"  ✗ Roll-rate requirement NOT met "
              f"({np.degrees(r.roll_rate):.2f} < {r.inputs.roll_req_deg:.1f} °/s)")
    print(f"  Aileron inboard y/(b/2) : {r.start_y_frac:.3f}")
    print(f"  Aileron outboard y/(b/2): {r.inputs.max_y_frac:.3f}")
    print(f"  Aileron span (per side) : {r.aileron_span:.3f}  m")
    print(f"  τ (effectiveness)       : {r.tau:.3f}")
    print(f"  cd0 (section, used)     : {r.cd0_section:.5f}")
    print(f"  Cl_δa                   : {r.cl_da:.4f}  1/rad")
    print(f"  Cl_p                    : {r.cl_p:.4f}  1/rad")
    print(f"  Steady roll rate        : {np.degrees(r.roll_rate):.2f}  °/s")


if __name__ == "__main__":
    from sizing import initial_sizing
    summary(run(initial_sizing.run()))
