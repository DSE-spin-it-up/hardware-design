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
    roll_rate_req_deg_s: float = 30.0 # required steady roll rate [deg/s]
    max_da_deg: float = 10.0         # max aileron deflection [deg]
    max_y_frac: float = 0.8          # outboard aileron edge, y/(b/2) [-]
    payload_lateral_angle_deg: float = 0.0  # unused; retained for API compat
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
    converged: bool        # True if roll-rate requirement met before hitting root
    roll_moment_payload: float   # roll moment from payload tension [-]
    payload_lateral_force: float # T_max used for payload criterion [N]
    payload_moment_arm: float    # vertical arm from payload attach point to CG [m]
    q_sizing: float              # dynamic pressure used for payload sizing [Pa]
    driving_constraint: str      # "roll_rate", "payload", or "both"
    cl_da_integral: float        # int c(y)*y dy over one aileron, physical units [m^3]
    cl_p_integral: float         # int c(y)*y^2 dy over one semi-span, physical units [m^4]
    S_ref: float                 # wing reference area used in derivatives [m^2]
    b_ref: float                 # wing span used in derivatives [m]
    payload_check_passed: bool   # True if roll_moment ≥ roll_moment_payload (or N/A)
    # Per-constraint inboard fractions (diagnostic)
    start_y_frac_roll_rate: float  # inboard edge from roll-rate search alone
    start_y_frac_payload: float    # inboard edge from payload search alone (NaN when N/A)


def chord_at_y_frac(c_root: float, lam: float):
    """Linear-taper chord as a function of y/(b/2)."""
    def f(y_frac: float) -> float:
        return c_root * (1 - (1 - lam) * y_frac)
    return f


def tau_from_ratio(c_aileron_to_c_wing: float) -> float:
    """Aileron effectiveness τ from chord ratio (polynomial empirical fit)."""
    if not (0.0 < c_aileron_to_c_wing < 0.7):
        raise ValueError(
            f"c_aileron/c_wing = {c_aileron_to_c_wing} is outside valid range (0, 0.7)."
        )
    return float(np.polyval([-6.624, 12.07, -8.292, 3.295, 0.004942], c_aileron_to_c_wing))


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
    integrand = lambda y: c_at_y_frac(y) * y * (b / 2) ** 2
    c_integral, _ = quad(integrand, start_y_frac, end_y_frac)
    return (2 * cl_alpha * tau) / (S_ref * b) * c_integral


def cl_da_integral(start_y_frac: float, end_y_frac: float, c_at_y_frac, b: float) -> float:
    """Physical one-side aileron integral int c(y)*y dy [m^3]."""
    integrand = lambda y: c_at_y_frac(y) * y * (b / 2) ** 2
    c_integral, _ = quad(integrand, start_y_frac, end_y_frac)
    return float(c_integral)


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


def cl_p_integral(c_at_y_frac, b: float) -> float:
    """Physical one-side roll-damping integral int c(y)*y^2 dy [m^4]."""
    integrand = lambda y: c_at_y_frac(y) * (y ** 2) * (b / 2) ** 3
    c_integral, _ = quad(integrand, 0.0, 1.0)
    return float(c_integral)


def _inboard_span_search(
    *,
    max_y_frac: float,
    step: float,
    c_at_y_frac,
    b: float,
    S_ref: float,
    cl_alpha: float,
    tau: float,
    criterion,          # callable(cl_da: float) -> bool  — True when requirement met
) -> tuple[float, float, bool]:
    """Sweep the inboard edge inward until *criterion* is satisfied.

    Returns
    -------
    start_y_frac : converged inboard edge (or 0.0 if root reached)
    cl_da        : Cl_δa at the returned geometry
    converged    : True when criterion was met before reaching the root
    """
    start_y_frac = max_y_frac
    cl_da = 0.0
    converged = False
    while start_y_frac > 0.0:
        start_y_frac -= step
        cl_da = compute_cl_da(
            start_y_frac, max_y_frac, c_at_y_frac, b, S_ref, cl_alpha, tau,
        )
        if criterion(cl_da):
            converged = True
            break
    return start_y_frac, cl_da, converged


def run(
    sizing: SizingResult,
    inputs: AileronInputs | None = None,
    polar: AirfoilPolar | None = None,
    y_cg: dict[str, float] | None = None,
    q_sizing: float | None = None,
    payload_max_tension: float | None = None,
) -> AileronResult:
    """Size the ailerons against two independent constraints and take the
    most conservative (largest) aileron span.

    Constraints
    -----------
    1. **Roll-rate** — steady roll rate ≥ roll_rate_req_deg_s at δ_a_max.
    2. **Payload roll moment** — Cl_δa·δ_a_max ≥ Cl_payload at δ_a_max.
       Active only when *y_cg* is supplied; otherwise skipped.

    The inboard edge that requires the *smaller* y/(b/2) value (i.e. the
    larger aileron span) becomes the design point.  ``driving_constraint``
    reports which constraint set the geometry: ``"roll_rate"``,
    ``"payload"``, or ``"both"`` when both converge to the same fraction
    (within one step).

    Parameters
    ----------
    sizing  : converged wing sizing result
    inputs  : AileronInputs overrides; defaults used when None
    polar   : wing airfoil polar for section Cd_p; fallback used when None
    y_cg    : vertical CG dict from compute_y_cg (keys: 'overall',
              'pvc_tube_bottom').  When None, constraint 2 is inactive.
    payload_max_tension : maximum cable tension [N]; when None, falls back
              to (m_payload / n_drones) * g.
    """
    if inputs is None:
        inputs = AileronInputs()
    i = inputs
    s = sizing
    q_size = s.q_cruise if q_sizing is None else q_sizing

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
    # Payload roll-moment target
    #
    # Cl_payload = (vertical_arm * T_max) / (q * Sw * b)
    # ------------------------------------------------------------------
    payload_moment_arm = 0.0
    payload_lateral_force = 0.0
    roll_moment_payload = 0.0

    if y_cg is not None:
        payload_moment_arm = abs(y_cg["overall"] - y_cg["pvc_tube_bottom"])
        n_drones = max(float(s.inputs.n_drones), 1.0)
        if payload_max_tension is not None:
            payload_lateral_force = float(payload_max_tension)
        else:
            payload_mass_per_drone = s.inputs.m_payload / n_drones
            payload_lateral_force = payload_mass_per_drone * 9.80665

        denom = q_size * s.Sw * s.inputs.b
        roll_moment_payload = (
            (payload_moment_arm * payload_lateral_force) / denom
            if denom > 0.0 else 0.0
        )

    roll_rate_required = np.radians(i.roll_rate_req_deg_s)

    # ------------------------------------------------------------------
    # Constraint 1 — roll rate
    # ------------------------------------------------------------------
    def roll_rate_criterion(cl_da: float) -> bool:
        P = abs(-(cl_da / cl_p) * max_da * (2 * s.inputs.V_cruise / s.inputs.b))
        return P >= roll_rate_required

    y_rr, cl_da_rr, conv_rr = _inboard_span_search(
        max_y_frac=i.max_y_frac,
        step=i.step,
        c_at_y_frac=c_at_y_frac,
        b=s.inputs.b,
        S_ref=s.Sw,
        cl_alpha=i.cl_alpha,
        tau=tau,
        criterion=roll_rate_criterion,
    )

    # ------------------------------------------------------------------
    # Constraint 2 — payload roll moment  (only when y_cg supplied)
    # ------------------------------------------------------------------
    payload_active = roll_moment_payload > 0.0

    if payload_active:
        def payload_criterion(cl_da: float) -> bool:
            return abs(cl_da * max_da) >= roll_moment_payload

        y_pl, cl_da_pl, conv_pl = _inboard_span_search(
            max_y_frac=i.max_y_frac,
            step=i.step,
            c_at_y_frac=c_at_y_frac,
            b=s.inputs.b,
            S_ref=s.Sw,
            cl_alpha=i.cl_alpha,
            tau=tau,
            criterion=payload_criterion,
        )
    else:
        y_pl, cl_da_pl, conv_pl = float("nan"), 0.0, False

    # ------------------------------------------------------------------
    # Select the most conservative (largest) aileron span
    # The smaller inboard y/(b/2) fraction means the aileron extends
    # further inboard → larger span → more authority.
    # ------------------------------------------------------------------
    if not payload_active:
        # Only roll-rate constraint is active
        start_y_frac = y_rr
        cl_da = cl_da_rr
        converged = conv_rr
        driving_constraint = "roll_rate"
    else:
        y_rr_eff = y_rr if conv_rr else 0.0
        y_pl_eff = y_pl if conv_pl else 0.0

        if not conv_rr and not conv_pl:
            # Neither converged — report the roll-rate result (primary constraint)
            start_y_frac = y_rr
            cl_da = cl_da_rr
            converged = False
            driving_constraint = "roll_rate"
        elif y_rr_eff == y_pl_eff:
            start_y_frac = y_rr_eff
            cl_da = compute_cl_da(
                start_y_frac, i.max_y_frac, c_at_y_frac,
                s.inputs.b, s.Sw, i.cl_alpha, tau,
            )
            converged = True
            driving_constraint = "both"
        elif y_rr_eff <= y_pl_eff:
            # Roll-rate needs the larger span
            start_y_frac = y_rr_eff
            cl_da = cl_da_rr if conv_rr else compute_cl_da(
                start_y_frac, i.max_y_frac, c_at_y_frac,
                s.inputs.b, s.Sw, i.cl_alpha, tau,
            )
            converged = conv_rr
            driving_constraint = "roll_rate"
        else:
            # Payload needs the larger span
            start_y_frac = y_pl_eff
            cl_da = cl_da_pl if conv_pl else compute_cl_da(
                start_y_frac, i.max_y_frac, c_at_y_frac,
                s.inputs.b, s.Sw, i.cl_alpha, tau,
            )
            converged = conv_pl
            driving_constraint = "payload"

    # Final derived quantities at the selected geometry
    roll_moment = abs(cl_da * max_da)
    P = abs(-(cl_da / cl_p) * max_da * (2 * s.inputs.V_cruise / s.inputs.b))
    aileron_span = (i.max_y_frac - start_y_frac) * (s.inputs.b / 2)
    cl_da_int = cl_da_integral(start_y_frac, i.max_y_frac, c_at_y_frac, s.inputs.b)
    cl_p_int = cl_p_integral(c_at_y_frac, s.inputs.b)

    payload_check_passed = roll_moment >= roll_moment_payload

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
        payload_lateral_force=payload_lateral_force,
        payload_moment_arm=payload_moment_arm,
        q_sizing=q_size,
        driving_constraint=driving_constraint,
        cl_da_integral=cl_da_int,
        cl_p_integral=cl_p_int,
        S_ref=s.Sw,
        b_ref=s.inputs.b,
        payload_check_passed=payload_check_passed,
        start_y_frac_roll_rate=y_rr,
        start_y_frac_payload=y_pl,
    )


def _summary_legacy(r: AileronResult) -> None:
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


def summary(r: AileronResult) -> None:
    roll_rate_req = r.inputs.roll_rate_req_deg_s
    roll_rate_deg_s = np.degrees(r.roll_rate)
    if r.converged:
        print(f"  Roll-rate requirement met ({roll_rate_deg_s:.2f} >= {roll_rate_req:.2f} deg/s)")
    else:
        print(f"  X Roll-rate requirement NOT met ({roll_rate_deg_s:.2f} < {roll_rate_req:.2f} deg/s)")

    # Payload constraint result
    if r.payload_moment_arm > 0.0 and r.roll_moment_payload > 0.0:
        margin = r.roll_moment - r.roll_moment_payload
        status = "PASS" if r.payload_check_passed else "FAIL"
        print(f"  Payload roll constraint     : {status} "
              f"(Cl_control {r.roll_moment:.4f} {'≥' if r.payload_check_passed else '<'} "
              f"Cl_payload {r.roll_moment_payload:.4f}, margin {margin:+.4f})")
    else:
        print(f"  Payload roll constraint     : N/A (no y_cg supplied)")

    print(f"  Active constraint           : {r.driving_constraint}")

    # Per-constraint inboard fractions (diagnostic)
    rr_str = f"{r.start_y_frac_roll_rate:.3f}"
    pl_str = (f"{r.start_y_frac_payload:.3f}"
              if not np.isnan(r.start_y_frac_payload) else "N/A")
    print(f"  Inboard y/(b/2) — roll-rate : {rr_str}")
    print(f"  Inboard y/(b/2) — payload   : {pl_str}")
    print(f"  Inboard y/(b/2) — selected  : {r.start_y_frac:.3f}  ← largest b_a/b")

    print(f"  Required roll rate          : {roll_rate_req:.2f}  deg/s")
    print(f"  Roll-moment (disturbance)   : {r.inputs.roll_moment_dist:.4f}  (check only)")
    print(f"  Roll-moment (payload)       : {r.roll_moment_payload:.4f}")
    print(f"  Payload roll margin         : {r.roll_moment - r.roll_moment_payload:+.4f}")
    print(f"  Payload lateral force (T_max): {r.payload_lateral_force:.3f}  N")
    print(f"  Payload moment arm          : {r.payload_moment_arm:.4f}  m")
    print(f"  Sizing dynamic pressure     : {r.q_sizing:.2f}  Pa")
    print(f"  Formula: Cl_payload = (arm * T_max) / (q * Sw * b)")
    print(f"  Aileron inboard y/(b/2)     : {r.start_y_frac:.3f}")
    print(f"  Aileron outboard y/(b/2)    : {r.inputs.max_y_frac:.3f}")
    print(f"  Aileron span (per side)     : {r.aileron_span:.3f}  m")
    print(f"  tau_a                       : {r.tau:.3f}")
    print(f"  cd0 section used in Cl_p    : {r.cd0_section:.5f}")
    print("  Sensitivity inputs:")
    print(f"    cl_alpha                  : {r.inputs.cl_alpha:.4f}  1/rad")
    print(f"    S_ref, b_ref              : {r.S_ref:.4f} m^2, {r.b_ref:.4f} m")
    print(f"    int aileron c*y dy        : {r.cl_da_integral:.6f}  m^3")
    print(f"    int semi-span c*y^2 dy    : {r.cl_p_integral:.6f}  m^4")
    print(f"    delta_a_max               : {r.inputs.max_da_deg:.2f}  deg")
    print(f"    roll_rate_req             : {roll_rate_req:.2f}  deg/s")
    print(f"    payload lateral force     : {r.payload_lateral_force:.3f}  N")
    print(f"    payload moment arm        : {r.payload_moment_arm:.4f}  m")
    print(f"    q_sizing                  : {r.q_sizing:.2f}  Pa")
    print("  Formulas:")
    print("    Cl_delta_a = 2*cl_alpha*tau_a/(S_ref*b_ref) * int_aileron(c*y dy)")
    print("    Cl_p       = -4*(cl_alpha+cd0_section)/(S_ref*b_ref^2) * int_semispan(c*y^2 dy)")
    print("    Cl_control = abs(Cl_delta_a * delta_a_max_rad)")
    print("    p_steady   = abs(-(Cl_delta_a/Cl_p)*delta_a_max_rad*2*V/b)")
    print("    T_max      = payload_max_tension  or  (m_payload/n_drones)*g  when unset")
    print("    Cl_payload = (arm * T_max) / (q * Sw * b)")
    print("    Geometry: min(y_rr, y_pl) → largest b_a/b (most conservative)")
    print(f"  Cl_delta_a                  : {r.cl_da:.4f}  1/rad")
    print(f"  Cl_p                        : {r.cl_p:.4f}  1/rad")
    print(f"  Roll-control moment         : {r.roll_moment:.4f}  (at delta_a_max)")
    print(f"  Steady roll rate            : {np.degrees(r.roll_rate):.2f}  deg/s")


if __name__ == "__main__":
    from sizing import wing
    summary(run(wing.run()))