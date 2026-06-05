"""Rod sizing for the wing and tail.

Both rods are sized closed-form for the thinnest CFRP wall that
simultaneously satisfies a tip-deflection limit and a compressive-stress
limit.

Wing has two rods per half-span:
  - Spar rod    : at the max-thickness location; carries all bending.
  - Aileron rod : at the aileron hinge (x/c = 1 - c_aileron/c_wing);
                  diameter limited by the thinner local section height.

The aileron hinge x/c comes from AileronResult so there is no hard-coded
chord fraction in this module.

V-tail has four rods total, two per V-tail plane, each of length bt/2:
  - Forward (spar) rod : at the max-thickness location of the tail airfoil.
  - Aft (ruddervator)  : at the ruddervator hinge
                         (x/c = 1 - c_ruddervator_to_c_tail);
                         sized identically to the wing aileron rod logic
                         but each rod carries F_tail/2 (quarter of total
                         tail load).

Tail rod: cantilever connecting the tail to the aileron-hinge anchor on the
wing, with the tail download as a point load at the tip. Length = L_boom
(aileron hinge → tail TE), strictly longer than the aero moment arm lh
since it extends past the tail AC to its trailing edge.

Tail rod torsion: the rudder hinge moment at cruise speed and maximum
deflection is transferred into the boom as a torque.  The tail rod is
checked for combined bending + torsion using the von Mises criterion.
If the torsion check governs, the diameter is increased accordingly.

References
----------
Bending / deflection / compressive stress:
    Standard Euler-Bernoulli beam theory (any structures textbook).

Torsional shear stress (Bredt-Batho, closed thin-walled section):
    τ = T / (2 · A_m · t),  A_m = π·(d/2)²
    Megson, "Aircraft Structures for Engineering Students", Ch. 17.

Polar moment of area for thin-walled tube:
    J = π·t·d³/4  (= 2·I for a circular section)
    Megson, Ch. 11.

Combined bending + torsion, von Mises criterion:
    σ_vm = sqrt(σ_b² + 3·τ²) ≤ σ_allow
    Megson, Ch. 11; Bruhn, "Analysis and Design of Flight Vehicle
    Structures", Section C2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import propulsion
from aerodynamics.airfoil_geometry import AirfoilGeometry
from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import FlightCondition, WingGeometry, solve_llt
from sizing.aileron import AileronResult
from sizing.wing import SizingResult
from structures.materials import CFRP
from propulsion.sizing import PropulsionResult


@dataclass
class RodInputs:
    material: CFRP = field(default_factory=CFRP)
    safety_factor: float = 1.2
    defl_max: float = 0.05            # [m] max tip deflection
    d_to_section_ratio: float = 0.8   # rod OD as a fraction of local section thickness
    CLt_max: float = 1.0              # tail max lift coefficient for tail-rod sizing
    tail_tc: float = 0.10             # tail-airfoil t/c for the geometric fit
    eta_h: float = 0.85               # tail dynamic pressure ratio q_t/q (from elevator config)
    eta_v: float = 0.85               # vertical tail dynamic pressure ratio q_v/q (from rudder config)
    t_spar: float = 0.00079375        # [m] minimum wall thickness of the spar rod
    t_control: float = 0.0016256      # [m] minimum wall thickness of the aileron rod
    t_control_ht: float = 0.0014      # [m] same as t_spar, thinner than t_control
    t_t: float = 0.00079375           # [m] minimum wall thickness of the tail rod
    # Ruddervator hinge chord fraction (analogous to c_aileron_to_c_wing).
    # x/c_hinge = 1 - c_ruddervator_to_c_tail
    c_ruddervator_to_c_tail: float = 0.35  # ruddervator chord / tail chord


@dataclass
class RodResult:
    inputs: RodInputs
    # Spar rod — at max-thickness x/c (one rod per half; total wing uses 2×)
    d_spar: float
    t_spar: float
    mass_spar: float
    defl_spar: float
    fail_mode_spar: str     # "deflection" | "compressive"
    # Aileron rod — at hinge x/c (one rod per half; total wing uses 2×)
    d_aileron: float
    t_control: float
    mass_aileron: float
    defl_aileron: float
    fail_mode_aileron: str
    # Tail rod (aileron hinge → tail TE, length = L_boom)
    d_t: float
    t_t: float
    mass_t: float
    defl_t: float
    fail_mode_t: str        # "deflection" | "compressive" | "torsion (von Mises)"
    eta_h: float
    eta_v: float
    d_spar_ht: float
    t_spar_ht: float
    mass_spar_ht: float
    defl_spar_ht: float
    fail_mode_spar_ht: str
    d_control_ht: float
    t_control_ht: float
    mass_control_ht: float
    defl_control_ht: float
    fail_mode_control_ht: str
    d_spar_vt: float
    t_spar_vt: float
    mass_spar_vt: float
    defl_spar_vt: float
    fail_mode_spar_vt: str
    d_control_vt: float
    t_control_vt: float
    mass_control_vt: float
    defl_control_vt: float
    fail_mode_control_vt: str
    f_n_wing: float = 0.0
    V_structural: float = 0.0
    q_structural: float = 0.0
    CL_h_structural: float = 0.0
    F_tail_structural: float = 0.0
    V_tail_structural: float = 0.0
    q_tail_structural: float = 0.0
    # Tail rod torsion results (populated by apply_torsion_check)
    tau_t: float = 0.0        # torsional shear stress in tail rod [Pa]
    sigma_vm_t: float = 0.0   # von Mises stress in tail rod [Pa]
    torsion_checked: bool = False
    # Geometric fit results — True when the structurally-required diameter
    # fits within the local airfoil section (d ≤ d_to_section_ratio × h_local).
    fits_spar: bool = True
    fits_aileron: bool = True
    fits_spar_ht: bool = True
    fits_control_ht: bool = True
    fits_spar_vt: bool = True
    fits_control_vt: bool = True
    # Local section heights at each rod placement [m]
    section_h_spar: float = 0.0
    section_h_aileron: float = 0.0
    section_h_spar_ht: float = 0.0
    section_h_control_ht: float = 0.0
    section_h_spar_vt: float = 0.0
    section_h_control_vt: float = 0.0

    # Convenience aliases so existing callers that use .d_w / .mass_w still work.
    @property
    def d_w(self) -> float:
        return self.d_spar

    @property
    def t_w(self) -> float:
        return self.t_spar

    @property
    def mass_w(self) -> float:
        return self.mass_spar

    @property
    def defl_w(self) -> float:
        return self.defl_spar

    @property
    def fail_mode_w(self) -> str:
        return self.fail_mode_spar


# ---------------------------------------------------------------------------
# Beam mechanics helpers
# ---------------------------------------------------------------------------

def _I_tube(t: float, d: float) -> float:
    """Thin-wall tube second moment of area: I = π·t·d³/8."""
    return np.pi * t * d ** 3 / 8


def _J_tube(t: float, d: float) -> float:
    """Thin-wall tube polar moment of area: J = π·t·d³/4 = 2·I.

    For a circular thin-walled tube J = 2·I (Megson Ch. 11).
    """
    return np.pi * t * d ** 3 / 4


def _d_for_defl_half_cantilever_udl(
    F_total: float, L_span: float, E: float, t: float, defl_max: float
) -> float:
    """Diameter for tip deflection ≤ defl_max (half-cantilever, UDL).

    δ = F·L³/(8·E·I); I = π·t·d³/8  →  d = (F·L³/(π·E·t·δ))^(1/3)
    """
    F, L = F_total / 2, L_span / 2
    return (F * L ** 3 / (np.pi * E * t * defl_max)) ** (1 / 3)


def _d_for_defl_cantilever_point(
    F: float, L: float, E: float, t: float, defl_max: float
) -> float:
    """Diameter for tip deflection ≤ defl_max (cantilever, point load).

    δ = F·L³/(3·E·I); I = π·t·d³/8  →  d = (8·F·L³/(3·π·E·t·δ))^(1/3)
    """
    return (8 * F * L ** 3 / (3 * np.pi * E * t * defl_max)) ** (1 / 3)


def _d_for_stress(M: float, sigma_lim: float, t: float) -> float:
    """Diameter so bending stress ≤ sigma_lim at outer fibre."""
    return np.sqrt(4 * M / (np.pi * sigma_lim * t))


def _defl_half_cantilever_udl(
    F_total: float, L_span: float, E: float, I: float
) -> float:
    F, L = F_total / 2, L_span / 2
    return F * L ** 3 / (8 * E * I)


def _defl_cantilever_point(F: float, L: float, E: float, I: float) -> float:
    return F * L ** 3 / (3 * E * I)


def _wing_natural_frequency(E: float, I: float, L: float, mass: float) -> float:
    """First bending mode natural frequency for a cantilever beam.

    f1 = beta1^2 / (2π) · sqrt(EI / (m'L^4)), where m' is mass per length.
    """
    if L <= 0 or mass <= 0:
        return 0.0
    beta1 = 1.875104068711961
    m_per_length = mass / L
    return beta1 ** 2 / (2 * np.pi) * np.sqrt(E * I / (m_per_length * L ** 4))


def _finite_tail_cl_limit_llt(sizing: SizingResult, tail_polar: AirfoilPolar) -> float:
    """Finite-tail CL limit from LLT at the first section Cl limit."""
    geom = WingGeometry(
        b=sizing.bh,
        S=sizing.Sh,
        taper=sizing.inputs.lam_t,
    )
    flight0 = FlightCondition(
        V_inf=sizing.inputs.V_cruise,
        rho=sizing.rho,
        alpha_root=0.0,
    )
    flight1 = FlightCondition(
        V_inf=sizing.inputs.V_cruise,
        rho=sizing.rho,
        alpha_root=1.0,
    )
    llt0 = solve_llt(geom, tail_polar, flight0)
    llt1 = solve_llt(geom, tail_polar, flight1)

    cl0 = llt0.Cl_local
    cl_slope = llt1.Cl_local - cl0
    cl_section_max = float(np.max(tail_polar.Cl))
    cl_section_min = float(np.min(tail_polar.Cl))

    positive_alpha = np.divide(
        cl_section_max - cl0,
        cl_slope,
        out=np.full_like(cl0, np.inf),
        where=cl_slope > 1.0e-12,
    )
    positive_alpha = positive_alpha[positive_alpha > 0.0]

    negative_alpha = np.divide(
        cl_section_min - cl0,
        cl_slope,
        out=np.full_like(cl0, -np.inf),
        where=cl_slope > 1.0e-12,
    )
    negative_alpha = negative_alpha[negative_alpha < 0.0]

    candidates: list[float] = []
    if positive_alpha.size:
        llt_pos = solve_llt(
            geom,
            tail_polar,
            FlightCondition(
                V_inf=sizing.inputs.V_cruise,
                rho=sizing.rho,
                alpha_root=float(np.min(positive_alpha)),
            ),
        )
        candidates.append(abs(llt_pos.CL))
    if negative_alpha.size:
        llt_neg = solve_llt(
            geom,
            tail_polar,
            FlightCondition(
                V_inf=sizing.inputs.V_cruise,
                rho=sizing.rho,
                alpha_root=float(np.max(negative_alpha)),
            ),
        )
        candidates.append(abs(llt_neg.CL))

    if not candidates:
        raise RuntimeError("Could not derive finite-tail CL limit from LLT.")
    return max(candidates)


def _tube_mass(length: float, d: float, t: float, rho: float) -> float:
    return ((d / 2) ** 2 - (d / 2 - t) ** 2) * np.pi * length * rho


def _check_wall(t: float, d: float, label: str) -> float:
    if t > d / 2:
        print(f"{label}: required wall thickness {t * 1000:.2f} mm exceeds rod radius "
              f"{d / 2 * 1000:.2f} mm — rod cannot satisfy criteria at this diameter. "
              f"Rod diameter is increased to {2 * t * 1000:.2f} mm")
        return 2 * t
    else:
        return d


def _check_geometric_fit(
    d: float,
    section_h: float,
    d_to_section_ratio: float,
    label: str,
) -> bool:
    """Return True if the rod fits within the airfoil section; print a warning if not.

    The rod outer diameter must satisfy d ≤ d_to_section_ratio × section_h.
    The check is informational — it does NOT alter d, because there is no
    structural remedy available at this stage (the section height is fixed by
    aerodynamic design).  The caller records the result in RodResult so that
    higher-level design loops can act on it.

    Parameters
    ----------
    d                  : structurally-required rod outer diameter [m]
    section_h          : local airfoil section height at the rod x/c [m]
    d_to_section_ratio : allowable fraction of section height (e.g. 0.80)
    label              : human-readable rod name for the warning message

    Returns
    -------
    True  → rod fits (d ≤ ratio × section_h)
    False → rod does NOT fit; a warning is printed
    """
    d_max_geo = d_to_section_ratio * section_h
    if d > d_max_geo:
        print(
            f"WARNING — {label}: rod OD {d * 1000:.2f} mm exceeds the geometric "
            f"limit of {d_max_geo * 1000:.2f} mm "
            f"({d_to_section_ratio * 100:.0f}% × section height "
            f"{section_h * 1000:.2f} mm).  "
            f"The airfoil section is too thin to house this rod; consider a "
            f"thicker airfoil, a larger chord, or accepting the structural "
            f"penalty of a reduced wall thickness."
        )
        return False
    return True


def _tau_bredt(T: float, d: float, t: float) -> float:
    """Torsional shear stress via Bredt-Batho for a closed thin-walled tube."""
    A_m = np.pi * (d / 2) ** 2
    return T / (2 * A_m * t)


def _sigma_bending(M: float, d: float, t: float) -> float:
    """Peak bending stress at outer fibre of thin-walled tube."""
    I = _I_tube(t, d)
    return M * (d / 2) / I


def _d_for_von_mises(
    M: float, T: float, sigma_allow: float, t: float
) -> float:
    """Minimum diameter so von Mises stress ≤ sigma_allow under combined
    bending moment M and torque T for a thin-walled circular tube."""
    numerator = np.sqrt(16 * M ** 2 + 12 * T ** 2)
    return (numerator / (np.pi * t * sigma_allow)) ** 0.5


# ---------------------------------------------------------------------------
# Torsion check — called after rudder sizing is complete
# ---------------------------------------------------------------------------

def apply_torsion_check(
    rod: RodResult,
    rudder_hinge_moment: float,
    bending_force: float,
    boom_length: float,
    safety_factor: float = 1.2,
) -> RodResult:
    """Check and if necessary upsize the tail rod for combined bending + torsion."""
    import dataclasses

    T   = abs(rudder_hinge_moment) * safety_factor
    M   = bending_force * boom_length * safety_factor
    t   = rod.t_t
    mat = rod.inputs.material

    d_vm = _d_for_von_mises(M, T, mat.s_c, t)
    d_vm = _check_wall(t, d_vm, "Tail rod (torsion)")

    if d_vm > rod.d_t:
        d_final   = d_vm
        fail_mode = "torsion (von Mises)"
    else:
        d_final   = rod.d_t
        fail_mode = rod.fail_mode_t

    tau      = _tau_bredt(T, d_final, t)
    sigma_b  = _sigma_bending(M, d_final, t)
    sigma_vm = np.sqrt(sigma_b ** 2 + 3 * tau ** 2)
    mass_t   = _tube_mass(boom_length, d_final, t, mat.rho)

    return dataclasses.replace(
        rod,
        d_t             = d_final,
        mass_t          = mass_t,
        fail_mode_t     = fail_mode,
        tau_t           = tau,
        sigma_vm_t      = sigma_vm,
        torsion_checked = True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    sizing: SizingResult,
    aileron: AileronResult,
    propulsion: PropulsionResult,
    airfoil_path: str,          # wing airfoil .dat file
    tail_airfoil_path: str,     # tail airfoil .dat file
    inputs: RodInputs | None = None,
    tail_polar: AirfoilPolar | None = None,
    rudder=None,                # RudderResult — enables physics-based VT sizing
    CLalphav_vt: float = 0.0,  # vertical tail lift-curve slope [1/rad]
) -> RodResult:
    if inputs is None:
        inputs = RodInputs()
    i = inputs
    s = sizing
    si = s.inputs
    mat = i.material
    sigma_lim = mat.s_c
    E = mat.E
    rho_mat = mat.rho

    # Load airfoil geometry once for local thickness queries.
    airfoil      = AirfoilGeometry(airfoil_path)
    tail_airfoil = AirfoilGeometry(tail_airfoil_path)

    # Total wing lift and span (shared by both wing rods). The structural
    # case uses the larger of climb/cruise, with the configured gust speed
    # added on top.
    V_structural = max(propulsion.V_climb, si.V_cruise) + si.gust_speed
    q_structural = 0.5 * s.rho * V_structural ** 2
    L_lift = s.CL_one_drone_failure * q_structural * s.Sw
    b_w = si.b

    # ------------------------------------------------------------------ #
    # Spar rod — at max-thickness x/c                                     #
    # ------------------------------------------------------------------ #
    tc_spar, xc_spar = airfoil.compute_maximum_thickness()   # (t/c, x/c)
    section_h_spar = s.c_root * tc_spar
    t_spar = i.t_spar

    M_spar = L_lift * b_w / 16   # half-cantilever UDL max bending moment

    d_spar_defl = _d_for_defl_half_cantilever_udl(L_lift, b_w, E, t_spar, i.defl_max)
    d_spar_comp = _d_for_stress(M_spar, sigma_lim, t_spar)
    if d_spar_defl >= d_spar_comp:
        d_spar, fail_spar = d_spar_defl, "deflection"
    else:
        d_spar, fail_spar = d_spar_comp, "compressive"
    d_spar    = _check_wall(t_spar, d_spar, "Spar rod")
    defl_spar = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_spar, d_spar))
    mass_spar = _tube_mass(b_w, d_spar, t_spar, rho_mat)
    f_n_wing  = _wing_natural_frequency(E, _I_tube(t_spar, d_spar), b_w / 2, mass_spar)

    fits_spar = _check_geometric_fit(
        d_spar, section_h_spar, i.d_to_section_ratio,
        f"Wing spar rod (x/c = {xc_spar:.3f})"
    )

    # ------------------------------------------------------------------ #
    # Aileron rod — at hinge x/c = 1 - c_aileron/c_wing                  #
    # ------------------------------------------------------------------ #
    x_hinge = 1.0 - aileron.inputs.c_aileron_to_c_wing
    tc_aileron, _, _ = airfoil.compute_thickness(x_hinge)
    section_h_aileron = s.c_root * tc_aileron
    t_control = i.t_control

    d_ail_defl = _d_for_defl_half_cantilever_udl(L_lift, b_w, E, t_control, i.defl_max)
    d_ail_comp = _d_for_stress(M_spar, sigma_lim, t_control)
    if d_ail_defl >= d_ail_comp:
        d_aileron, fail_aileron = d_ail_defl, "deflection"
    else:
        d_aileron, fail_aileron = d_ail_comp, "compressive"
    d_aileron    = _check_wall(t_control, d_aileron, "Aileron rod")
    defl_aileron = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_control, d_aileron))
    mass_aileron = _tube_mass(b_w, d_aileron, t_control, rho_mat)

    fits_aileron = _check_geometric_fit(
        d_aileron, section_h_aileron, i.d_to_section_ratio,
        f"Wing aileron rod (x/c = {x_hinge:.3f})"
    )

    # ------------------------------------------------------------------ #
    # Tail rod (aileron hinge → tail TE, cantilever point load)           #
    # ------------------------------------------------------------------ #
    if tail_polar is not None:
        CL_h = _finite_tail_cl_limit_llt(s, tail_polar)
    else:
        CL_h = abs(-0.35 * s.inputs.ARt ** (1.0 / 3.0))

    # eta_h is the dynamic pressure ratio q_t/q (read from elevator config).
    # V_tail_structural is derived from it for reporting purposes only.
    q_tail_structural = q_structural * i.eta_h
    V_tail_structural = V_structural * np.sqrt(i.eta_h)
    F_tail = s.Sh * CL_h * q_tail_structural * i.safety_factor
    L_t    = s.L_boom
    M_t    = F_tail * L_t
    t_t    = i.t_t

    d_t_defl = _d_for_defl_cantilever_point(F_tail, L_t, E, t_t, i.defl_max)
    d_t_comp = _d_for_stress(M_t, sigma_lim, t_t)
    if d_t_defl >= d_t_comp:
        d_t, fail_t = d_t_defl, "deflection"
    else:
        d_t, fail_t = d_t_comp, "compressive"
    d_t    = _check_wall(t_t, d_t, "Tail rod")
    defl_t = _defl_cantilever_point(F_tail, L_t, E, _I_tube(t_t, d_t))
    mass_t = _tube_mass(L_t, d_t, t_t, rho_mat)
    # Tail (boom) rod is a circular tube in free air — no airfoil section
    # constraint applies.  No geometric fit check here.

    # ------------------------------------------------------------------ #
    # Horizontal tail — spar rod                                          #
    # ------------------------------------------------------------------ #
    tc_spar_ht, xc_spar_ht = tail_airfoil.compute_maximum_thickness()
    # Use the root chord of the horizontal tail for the section height.
    # bh is the full span; assuming a symmetric tail the root chord is Sh/bh
    # for an untapered surface, or c_root_ht if available on SizingResult.
    c_root_ht = getattr(s, "c_root_ht", s.Sh / s.bh)
    section_h_spar_ht = c_root_ht * tc_spar_ht

    L_ht      = s.bh
    F_ht_rod  = F_tail / 2
    t_spar_ht = i.t_spar

    M_spar_ht = F_ht_rod * L_ht / 16

    d_spar_defl_ht = _d_for_defl_half_cantilever_udl(F_ht_rod, L_ht, E, t_spar_ht, i.defl_max)
    d_spar_comp_ht = _d_for_stress(M_spar_ht, sigma_lim, t_spar_ht)
    if d_spar_defl_ht >= d_spar_comp_ht:
        d_spar_ht, fail_spar_ht = d_spar_defl_ht, "deflection"
    else:
        d_spar_ht, fail_spar_ht = d_spar_comp_ht, "compressive"
    d_spar_ht    = _check_wall(t_spar_ht, d_spar_ht, "Spar rod horizontal tail")
    defl_spar_ht = _defl_half_cantilever_udl(F_ht_rod, L_ht, E, _I_tube(t_spar_ht, d_spar_ht))
    mass_spar_ht = _tube_mass(L_ht, d_spar_ht, t_spar_ht, rho_mat)

    fits_spar_ht = _check_geometric_fit(
        d_spar_ht, section_h_spar_ht, i.d_to_section_ratio,
        f"HT spar rod (x/c = {xc_spar_ht:.3f})"
    )

    # ------------------------------------------------------------------ #
    # Horizontal tail — elevator rod                                      #
    # ------------------------------------------------------------------ #
    x_hinge_ht = 1.0 - i.c_ruddervator_to_c_tail
    tc_control_ht, _, _ = tail_airfoil.compute_thickness(x_hinge_ht)
    section_h_control_ht = c_root_ht * tc_control_ht

    t_control_ht = i.t_control_ht

    M_control_ht = F_ht_rod * L_ht / 16

    d_control_defl_ht = _d_for_defl_half_cantilever_udl(F_ht_rod, L_ht, E, t_control_ht, i.defl_max)
    d_control_comp_ht = _d_for_stress(M_control_ht, sigma_lim, t_control_ht)
    if d_control_defl_ht >= d_control_comp_ht:
        d_control_ht, fail_control_ht = d_control_defl_ht, "deflection"
    else:
        d_control_ht, fail_control_ht = d_control_comp_ht, "compressive"
    d_control_ht    = _check_wall(t_control_ht, d_control_ht, "Elevator rod horizontal tail")
    defl_control_ht = _defl_half_cantilever_udl(F_ht_rod, L_ht, E, _I_tube(t_control_ht, d_control_ht))
    mass_control_ht = _tube_mass(L_ht, d_control_ht, t_control_ht, rho_mat)

    fits_control_ht = _check_geometric_fit(
        d_control_ht, section_h_control_ht, i.d_to_section_ratio,
        f"HT elevator rod (x/c = {x_hinge_ht:.3f})"
    )

    # ------------------------------------------------------------------ #
    # Vertical tail — spar rod and rudder rod                             #
    # ------------------------------------------------------------------ #
    tc_spar_vt, xc_spar_vt = tail_airfoil.compute_maximum_thickness()
    x_hinge_vt = 1.0 - i.c_ruddervator_to_c_tail
    tc_control_vt, _, _ = tail_airfoil.compute_thickness(x_hinge_vt)
    # Root chord of the vertical tail: Sv / bv (single-surface)
    c_root_vt = getattr(s, "c_root_vt", s.Sv / s.bv)
    section_h_spar_vt    = c_root_vt * tc_spar_vt
    section_h_control_vt = c_root_vt * tc_control_vt

    L_vt = s.bv

    if rudder is not None and CLalphav_vt != 0.0:
        # ---- loads ----
        q_vt        = q_structural * i.eta_v
        F_fin       = 0.5 * q_vt * s.Sv * CLalphav_vt * rudder.beta_gust
        F_thrust_vt = 0.5 * propulsion.thrust_cruise_per_prop

        # ---- spar rod ----
        M_thrust_spar = F_thrust_vt * L_vt / 8
        M_fin_spar    = F_fin       * L_vt / 8
        M_spar_vt     = np.sqrt(M_thrust_spar ** 2 + M_fin_spar ** 2)
        t_spar_vt     = i.t_spar

        d_spar_defl_vt = _d_for_defl_half_cantilever_udl(
            F_thrust_vt + F_fin, L_vt, E, t_spar_vt, i.defl_max
        )
        d_spar_comp_vt = _d_for_stress(M_spar_vt, sigma_lim, t_spar_vt)
        if d_spar_defl_vt >= d_spar_comp_vt:
            d_spar_vt, fail_spar_vt = d_spar_defl_vt, "deflection"
        else:
            d_spar_vt, fail_spar_vt = d_spar_comp_vt, "compressive"
        d_spar_vt    = _check_wall(t_spar_vt, d_spar_vt, "Spar rod vertical tail")
        defl_spar_vt = _defl_half_cantilever_udl(
            F_thrust_vt + F_fin, L_vt, E, _I_tube(t_spar_vt, d_spar_vt)
        )
        mass_spar_vt = _tube_mass(L_vt, d_spar_vt, t_spar_vt, rho_mat)

        # ---- rudder rod ----
        bR_bV         = rudder.geometry.bR_bV
        F_rudder      = 0.5 * q_vt * s.Sv * CLalphav_vt * (
            rudder.beta_gust + rudder.tau_r * rudder.delta_R * bR_bV
        )
        F_thrust_vt_r = 0.5 * propulsion.thrust_cruise_per_prop
        T_rudder      = abs(rudder.hinge_moment.H)
        M_thrust_ctrl = F_thrust_vt_r * L_vt / 8
        M_rudder_ctrl = F_rudder      * L_vt / 8
        M_control_vt  = np.sqrt(M_thrust_ctrl ** 2 + M_rudder_ctrl ** 2)
        t_control_vt  = i.t_control

        d_control_defl_vt = _d_for_defl_half_cantilever_udl(
            F_thrust_vt_r + F_rudder, L_vt, E, t_control_vt, i.defl_max
        )
        d_control_vm_vt = _d_for_von_mises(M_control_vt, T_rudder, sigma_lim, t_control_vt)
        if d_control_defl_vt >= d_control_vm_vt:
            d_control_vt, fail_control_vt = d_control_defl_vt, "deflection"
        else:
            d_control_vt, fail_control_vt = d_control_vm_vt, "von Mises (bending + torsion)"
        d_control_vt    = _check_wall(t_control_vt, d_control_vt, "Rudder rod vertical tail")
        defl_control_vt = _defl_half_cantilever_udl(
            F_thrust_vt_r + F_rudder, L_vt, E, _I_tube(t_control_vt, d_control_vt)
        )
        mass_control_vt = _tube_mass(L_vt, d_control_vt, t_control_vt, rho_mat)

    else:
        # ---- fallback: thrust load only ----
        q_vt_fallback = q_structural * i.eta_v
        F_vt_rod  = propulsion.thrust_cruise_per_prop
        t_spar_vt = i.t_spar

        M_spar_vt = F_vt_rod * L_vt / 16

        d_spar_defl_vt = _d_for_defl_half_cantilever_udl(F_vt_rod, L_vt, E, t_spar_vt, i.defl_max)
        d_spar_comp_vt = _d_for_stress(M_spar_vt, sigma_lim, t_spar_vt)
        if d_spar_defl_vt >= d_spar_comp_vt:
            d_spar_vt, fail_spar_vt = d_spar_defl_vt, "deflection"
        else:
            d_spar_vt, fail_spar_vt = d_spar_comp_vt, "compressive"
        d_spar_vt    = _check_wall(t_spar_vt, d_spar_vt, "Spar rod vertical tail")
        defl_spar_vt = _defl_half_cantilever_udl(F_vt_rod, L_vt, E, _I_tube(t_spar_vt, d_spar_vt))
        mass_spar_vt = _tube_mass(L_vt, d_spar_vt, t_spar_vt, rho_mat)

        t_control_vt = i.t_control

        M_control_vt = F_vt_rod * L_vt / 16

        d_control_defl_vt = _d_for_defl_half_cantilever_udl(F_vt_rod, L_vt, E, t_control_vt, i.defl_max)
        d_control_comp_vt = _d_for_stress(M_control_vt, sigma_lim, t_control_vt)
        if d_control_defl_vt >= d_control_comp_vt:
            d_control_vt, fail_control_vt = d_control_defl_vt, "deflection"
        else:
            d_control_vt, fail_control_vt = d_control_comp_vt, "compressive"
        d_control_vt    = _check_wall(t_control_vt, d_control_vt, "Rudder rod vertical tail")
        defl_control_vt = _defl_half_cantilever_udl(F_vt_rod, L_vt, E, _I_tube(t_control_vt, d_control_vt))
        mass_control_vt = _tube_mass(L_vt, d_control_vt, t_control_vt, rho_mat)

    # Geometric fit checks — vertical tail (same airfoil for HT and VT)
    fits_spar_vt = _check_geometric_fit(
        d_spar_vt, section_h_spar_vt, i.d_to_section_ratio,
        f"VT spar rod (x/c = {xc_spar_vt:.3f})"
    )
    fits_control_vt = _check_geometric_fit(
        d_control_vt, section_h_control_vt, i.d_to_section_ratio,
        f"VT rudder rod (x/c = {x_hinge_vt:.3f})"
    )

    return RodResult(
        inputs=inputs,
        d_spar=d_spar, t_spar=t_spar, mass_spar=mass_spar,
        defl_spar=defl_spar, fail_mode_spar=fail_spar,
        d_aileron=d_aileron, t_control=t_control, mass_aileron=mass_aileron,
        defl_aileron=defl_aileron, fail_mode_aileron=fail_aileron,
        d_t=d_t, t_t=t_t, mass_t=mass_t, defl_t=defl_t, fail_mode_t=fail_t,
        f_n_wing=f_n_wing,
        V_structural=V_structural,
        q_structural=q_structural,
        CL_h_structural=CL_h,
        F_tail_structural=F_tail,
        V_tail_structural=V_tail_structural,
        q_tail_structural=q_tail_structural,
        eta_h=i.eta_h,
        eta_v=i.eta_v,
        d_spar_ht=d_spar_ht, t_spar_ht=t_spar_ht, mass_spar_ht=mass_spar_ht,
        defl_spar_ht=defl_spar_ht, fail_mode_spar_ht=fail_spar_ht,
        d_control_ht=d_control_ht, t_control_ht=t_control_ht, mass_control_ht=mass_control_ht,
        defl_control_ht=defl_control_ht, fail_mode_control_ht=fail_control_ht,
        d_spar_vt=d_spar_vt, t_spar_vt=t_spar_vt, mass_spar_vt=mass_spar_vt,
        defl_spar_vt=defl_spar_vt, fail_mode_spar_vt=fail_spar_vt,
        d_control_vt=d_control_vt, t_control_vt=t_control_vt, mass_control_vt=mass_control_vt,
        defl_control_vt=defl_control_vt, fail_mode_control_vt=fail_control_vt,
        # geometric fit flags
        fits_spar=fits_spar,
        fits_aileron=fits_aileron,
        fits_spar_ht=fits_spar_ht,
        fits_control_ht=fits_control_ht,
        fits_spar_vt=fits_spar_vt,
        fits_control_vt=fits_control_vt,
        # section heights
        section_h_spar=section_h_spar,
        section_h_aileron=section_h_aileron,
        section_h_spar_ht=section_h_spar_ht,
        section_h_control_ht=section_h_control_ht,
        section_h_spar_vt=section_h_spar_vt,
        section_h_control_vt=section_h_control_vt,
    )


def _fit_marker(fits: bool) -> str:
    return "OK" if fits else "*** DOES NOT FIT ***"


def summary(r: RodResult) -> None:
    print("\n--- Spar rod (one of two) ---")
    print(f"  Structural speed      : {r.V_structural:.2f}  m/s")
    print(f"  Structural q          : {r.q_structural:.2f}  Pa")
    print(f"  Outer diameter         : {r.d_spar * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_spar * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_spar * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_spar}")
    print(f"  Mass (each)            : {r.mass_spar:.3f}  kg")
    print(f"  First bending frequency: {r.f_n_wing:.2f}  Hz")
    print(f"  Section height at x/c  : {r.section_h_spar * 1000:.2f}  mm  →  geometric fit: {_fit_marker(r.fits_spar)}")

    print("\n--- Aileron rod (two of two) ---")
    print(f"  Outer diameter         : {r.d_aileron * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_control * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_aileron * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_aileron}")
    print(f"  Mass (each)            : {r.mass_aileron:.3f}  kg")
    print(f"  Section height at x/c  : {r.section_h_aileron * 1000:.2f}  mm  →  geometric fit: {_fit_marker(r.fits_aileron)}")

    print("\n--- Tail rod (aileron hinge → tail TE) ---")
    print(f"  Tail dynamic pressure ratio (η_h) : {r.eta_h:.3f}")
    print(f"  Structural tail speed  : {r.V_tail_structural:.2f}  m/s")
    print(f"  Structural tail q      : {r.q_tail_structural:.2f}  Pa")
    print(f"  Structural CL_h        : {r.CL_h_structural:.3f}")
    print(f"  Structural tail force  : {r.F_tail_structural:.2f}  N")
    print(f"  Outer diameter         : {r.d_t * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_t * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_t * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_t}")
    print(f"  Mass                   : {r.mass_t:.3f}  kg")
    if r.torsion_checked:
        print(f"  Torsional shear τ      : {r.tau_t / 1e6:.2f}  MPa")
        print(f"  Von Mises stress       : {r.sigma_vm_t / 1e6:.2f}  MPa")

    print("\n--- Spar rod horizontal tail (one of two) ---")
    print(f"  Outer diameter         : {r.d_spar_ht * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_spar_ht * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_spar_ht * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_spar_ht}")
    print(f"  Mass (each)            : {r.mass_spar_ht:.3f}  kg")
    print(f"  Section height at x/c  : {r.section_h_spar_ht * 1000:.2f}  mm  →  geometric fit: {_fit_marker(r.fits_spar_ht)}")

    print("\n--- Elevator rod (two of two) ---")
    print(f"  Outer diameter         : {r.d_control_ht * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_control_ht * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_control_ht * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_control_ht}")
    print(f"  Mass (each)            : {r.mass_control_ht:.3f}  kg")
    print(f"  Section height at x/c  : {r.section_h_control_ht * 1000:.2f}  mm  →  geometric fit: {_fit_marker(r.fits_control_ht)}")

    print("\n--- Spar rod vertical tail (one of two) ---")
    print(f"  VT dynamic pressure ratio (η_v) : {r.eta_v:.3f}")
    print(f"  Outer diameter         : {r.d_spar_vt * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_spar_vt * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_spar_vt * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_spar_vt}")
    print(f"  Mass (each)            : {r.mass_spar_vt:.3f}  kg")
    print(f"  Section height at x/c  : {r.section_h_spar_vt * 1000:.2f}  mm  →  geometric fit: {_fit_marker(r.fits_spar_vt)}")

    print("\n--- Rudder rod (two of two) ---")
    print(f"  Outer diameter         : {r.d_control_vt * 1000:.2f}  mm")
    print(f"  Wall thickness         : {r.t_control_vt * 1000:.2f}  mm")
    print(f"  Tip deflection         : {r.defl_control_vt * 1000:.2f}  mm")
    print(f"  Sizing criterion       : {r.fail_mode_control_vt}")
    print(f"  Mass (each)            : {r.mass_control_vt:.3f}  kg")
    print(f"  Section height at x/c  : {r.section_h_control_vt * 1000:.2f}  mm  →  geometric fit: {_fit_marker(r.fits_control_vt)}")

    # Summary line — flag overall pass/fail
    all_fit = all([
        r.fits_spar, r.fits_aileron,
        r.fits_spar_ht, r.fits_control_ht,
        r.fits_spar_vt, r.fits_control_vt,
    ])
    print()
    if all_fit:
        print("  ✓ All rods fit within their respective airfoil sections.")
    else:
        print("  ✗ One or more rods exceed their airfoil section envelope — see warnings above.")


if __name__ == "__main__":
    from sizing import aileron, wing
    from sizing.wing import SizingInputs
    s = wing.run(SizingInputs())
    a = aileron.run(s)
    p = propulsion.run(s)
    summary(run(s, a, p, "airfoils/MH112.dat", "airfoils/NACA0010.dat"))