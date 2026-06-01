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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import propulsion
from aerodynamics.airfoil_geometry import AirfoilGeometry
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
    Vh_V: float = 0.85                # V_tail/V_cruise — used for tail download
    t_spar: float = 0.00079375        # [m] minimum wall thickness of the spar rod
    t_control: float = 0.0016256      # [m] minimum wall thickness of the aileron rod
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
    fail_mode_t: str
    Vh_V: float
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


def _tube_mass(length: float, d: float, t: float, rho: float) -> float:
    return ((d / 2) ** 2 - (d / 2 - t) ** 2) * np.pi * length * rho


def _check_wall(t: float, d: float, label: str) -> None:
    if t > d / 2:
        print(f"{label}: required wall thickness {t * 1000:.2f} mm exceeds rod radius "
        f"{d / 2 * 1000:.2f} mm — rod cannot satisfy criteria at this diameter. Rod diameter is increased to {2 * t * 1000:.2f} mm"
        )
        return 2 * t
    else:
        return d


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
    airfoil = AirfoilGeometry(airfoil_path)
    tail_airfoil = AirfoilGeometry(tail_airfoil_path)

    # Total wing lift and span (shared by both wing rods).
    L_lift = ((si.m_drone_empty + si.m_payload) / (si.n_drones-1)+si.m_drone_empty) * 9.81  # [N]
    b_w = si.b

    # ------------------------------------------------------------------ #
    # Spar rod — at max-thickness x/c                                     #
    # ------------------------------------------------------------------ #
    tc_spar, _ = airfoil.compute_maximum_thickness()   # (t/c, x/c)
    section_h_spar = s.c_root * tc_spar
    d_spar_max = i.d_to_section_ratio * section_h_spar
    t_spar = i.t_spar

    M_spar = L_lift * b_w / 16             # half-cantilever UDL max bending moment

    d_spar_defl = _d_for_defl_half_cantilever_udl(L_lift, b_w, E, t_spar, i.defl_max)
    d_spar_comp = _d_for_stress(M_spar, sigma_lim, t_spar)
    if d_spar_defl >= d_spar_comp:
        d_spar, fail_spar = d_spar_defl, "deflection"
    else:
        d_spar, fail_spar = d_spar_comp, "compressive"
    d_spar = _check_wall(t_spar, d_spar, "Spar rod")
    defl_spar = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_spar, d_spar))
    mass_spar = _tube_mass(b_w, d_spar, t_spar, rho_mat)
    f_n_wing = _wing_natural_frequency(E, _I_tube(t_spar, d_spar), b_w / 2, mass_spar)

    # ------------------------------------------------------------------ #
    # Aileron rod — at hinge x/c = 1 - c_aileron/c_wing                  #
    # ------------------------------------------------------------------ #
    x_hinge = 1.0 - aileron.inputs.c_aileron_to_c_wing
    tc_aileron, _, _ = airfoil.compute_thickness(x_hinge)   # local t/c at hinge
    section_h_aileron = s.c_root * tc_aileron
    d_aileron = i.d_to_section_ratio * section_h_aileron
    t_control = i.t_control

    d_ail_defl = _d_for_defl_half_cantilever_udl(L_lift, b_w, E, t_control, i.defl_max)
    d_ail_comp = _d_for_stress(M_spar, sigma_lim, t_control)
    if d_ail_defl >= d_ail_comp:
        d_aileron, fail_aileron = d_ail_defl, "deflection"
    else:
        d_aileron, fail_aileron = d_ail_comp, "compressive"
    d_aileron = _check_wall(t_control, d_aileron, "Aileron rod")
    defl_aileron = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_control, d_aileron))
    mass_aileron = _tube_mass(b_w, d_aileron, t_control, rho_mat)

    # ------------------------------------------------------------------ #
    # Tail rod (aileron hinge → tail TE, cantilever point load)           #
    # ------------------------------------------------------------------ #
    CL_h = abs(-0.35 * s.inputs.ARt ** (1.0 / 3.0))

    F_tail = s.Sh * CL_h * s.q_cruise * i.Vh_V * i.safety_factor  # tail download [N]
    L_t = s.L_boom
    M_t = F_tail * L_t
    t_t = i.t_t

    d_t_defl = _d_for_defl_cantilever_point(F_tail, L_t, E, t_t, i.defl_max)
    d_t_comp = _d_for_stress(M_t, sigma_lim, t_t)
    if d_t_defl >= d_t_comp:
        d_t, fail_t = d_t_defl, "deflection"
    else:
        d_t, fail_t = d_t_comp, "compressive"
    d_t = _check_wall(t_t, d_t, "Tail rod")
    defl_t = _defl_cantilever_point(F_tail, L_t, E, _I_tube(t_t, d_t))
    mass_t = _tube_mass(L_t, d_t, t_t, rho_mat)

    # --- horizontal tail spar rod sizing
    L_ht = s.bh                          
    F_ht_rod = F_tail / 2                     
    t_spar_ht = i.t_spar

    M_spar_ht = F_ht_rod * L_ht / 16  # half-cantilever UDL max bending moment

    d_spar_defl_ht = _d_for_defl_half_cantilever_udl(F_ht_rod, L_ht, E, t_spar_ht, i.defl_max)
    d_spar_comp_ht = _d_for_stress(M_spar_ht, sigma_lim, t_spar_ht)
    if d_spar_defl_ht >= d_spar_comp_ht:
        d_spar_ht, fail_spar_ht = d_spar_defl_ht, "deflection"
    else:
        d_spar_ht, fail_spar_ht = d_spar_comp_ht, "compressive"
    d_spar_ht = _check_wall(t_spar_ht, d_spar_ht, "Spar rod horizontal tail")
    defl_spar_ht = _defl_half_cantilever_udl(F_ht_rod, L_ht, E, _I_tube(t_spar_ht, d_spar_ht))
    mass_spar_ht = _tube_mass(L_ht, d_spar_ht, t_spar_ht, rho_mat)

    # --- horizontal tail elevator rod sizing
    t_control_ht = i.t_control

    M_control_ht = F_ht_rod * L_ht / 16  # half-cantilever UDL max bending moment

    d_control_defl_ht = _d_for_defl_half_cantilever_udl(F_ht_rod, L_ht, E, t_control_ht, i.defl_max)
    d_control_comp_ht = _d_for_stress(M_control_ht, sigma_lim, t_control_ht)
    if d_control_defl_ht >= d_control_comp_ht:
        d_control_ht, fail_control_ht = d_control_defl_ht, "deflection"
    else:
        d_control_ht, fail_control_ht = d_control_comp_ht, "compressive"
    d_control_ht = _check_wall(t_control_ht, d_control_ht, "Elevator rod horizontal tail")
    defl_control_ht = _defl_half_cantilever_udl(F_ht_rod, L_ht, E, _I_tube(t_control_ht, d_control_ht))
    mass_control_ht = _tube_mass(L_ht, d_control_ht, t_control_ht, rho_mat)

    # --- vertical tail spar rod sizing
    L_vt = s.bv * 2
    F_vt_rod = propulsion.thrust_cruise_per_prop
    t_spar_vt = i.t_spar

    M_spar_vt = F_vt_rod * L_vt / 16  # half-cantilever UDL max bending moment

    d_spar_defl_vt = _d_for_defl_half_cantilever_udl(F_vt_rod, L_vt, E, t_spar_vt, i.defl_max)
    d_spar_comp_vt = _d_for_stress(M_spar_vt, sigma_lim, t_spar_vt)
    if d_spar_defl_vt >= d_spar_comp_vt:
        d_spar_vt, fail_spar_vt = d_spar_defl_vt, "deflection"
    else:
        d_spar_vt, fail_spar_vt = d_spar_comp_vt, "compressive"
    d_spar_vt = _check_wall(t_spar_vt, d_spar_vt, "Spar rod vertical tail")
    defl_spar_vt = _defl_half_cantilever_udl(F_vt_rod, L_vt, E, _I_tube(t_spar_vt, d_spar_vt))
    mass_spar_vt = _tube_mass(L_vt, d_spar_vt, t_spar_vt, rho_mat)

    # --- vertical tail rudder rod sizing
    t_control_vt = i.t_control

    M_control_vt = F_vt_rod * L_vt / 16  # half-cantilever UDL max bending moment

    d_control_defl_vt = _d_for_defl_half_cantilever_udl(F_vt_rod, L_vt, E, t_control_vt, i.defl_max)
    d_control_comp_vt = _d_for_stress(M_control_vt, sigma_lim, t_control_vt)
    if d_control_defl_vt >= d_control_comp_vt:
        d_control_vt, fail_control_vt = d_control_defl_vt, "deflection"
    else:
        d_control_vt, fail_control_vt = d_control_comp_vt, "compressive"
    d_control_vt = _check_wall(t_control_vt, d_control_vt, "Rudder rod horizontal tail")
    defl_control_vt = _defl_half_cantilever_udl(F_vt_rod, L_vt, E, _I_tube(t_control_vt, d_control_vt))
    mass_control_vt = _tube_mass(L_vt, d_control_vt, t_control_vt, rho_mat)
    
    

    return RodResult(
        inputs=inputs,
        d_spar=d_spar, t_spar=t_spar, mass_spar=mass_spar,
        defl_spar=defl_spar, fail_mode_spar=fail_spar,
        d_aileron=d_aileron, t_control=t_control, mass_aileron=mass_aileron,
        defl_aileron=defl_aileron, fail_mode_aileron=fail_aileron,
        d_t=d_t, t_t=t_t, mass_t=mass_t, defl_t=defl_t, fail_mode_t=fail_t,
        f_n_wing=f_n_wing,
        Vh_V=i.Vh_V,
        d_spar_ht=d_spar_ht, t_spar_ht=t_spar_ht, mass_spar_ht=mass_spar_ht,
        defl_spar_ht=defl_spar_ht, fail_mode_spar_ht=fail_spar_ht,
        d_control_ht=d_control_ht, t_control_ht=t_control_ht, mass_control_ht=mass_control_ht,
        defl_control_ht=defl_control_ht, fail_mode_control_ht=fail_control_ht,
        d_spar_vt=d_spar_vt, t_spar_vt=t_spar_vt, mass_spar_vt=mass_spar_vt,
        defl_spar_vt=defl_spar_vt, fail_mode_spar_vt=fail_spar_vt,
        d_control_vt=d_control_vt, t_control_vt=t_control_vt, mass_control_vt=mass_control_vt,
        defl_control_vt=defl_control_vt, fail_mode_control_vt=fail_control_vt
    )


def summary(r: RodResult) -> None:
    print("\n--- Spar rod (one of two) ---")
    print(f"  Outer diameter       : {r.d_spar * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_spar * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_spar * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_spar}")
    print(f"  Mass (each)          : {r.mass_spar:.3f}  kg")
    print(f"  First bending frequency: {r.f_n_wing:.2f}  Hz")

    print("\n--- Aileron rod (two of two) ---")
    print(f"  Outer diameter       : {r.d_aileron * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_control * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_aileron * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_aileron}")
    print(f"  Mass (each)          : {r.mass_aileron:.3f}  kg")

    print("\n--- Tail rod (aileron hinge → tail TE) ---")
    print(f"  Outer diameter       : {r.d_t * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_t * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_t * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_t}")
    print(f"  Mass                 : {r.mass_t:.3f}  kg")

    print("\n--- Spar rod horizontal tail (one of two) ---")
    print(f"  Outer diameter       : {r.d_spar_ht * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_spar_ht * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_spar_ht * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_spar_ht}")
    print(f"  Mass (each)          : {r.mass_spar_ht:.3f}  kg")

    print("\n--- Elevator rod (two of two) ---")
    print(f"  Outer diameter       : {r.d_control_ht * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_control_ht * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_control_ht * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_control_ht}")
    print(f"  Mass (each)          : {r.mass_control_ht:.3f}  kg")

    print("\n--- Spar rod vertical tail (one of two) ---")
    print(f"  Outer diameter       : {r.d_spar_vt * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_spar_vt * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_spar_vt * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_spar_vt}")
    print(f"  Mass (each)          : {r.mass_spar_vt:.3f}  kg")

    print("\n--- Rudder rod (two of two) ---")
    print(f"  Outer diameter       : {r.d_control_vt * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_control_vt * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_control_vt * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_control_vt}")
    print(f"  Mass (each)          : {r.mass_control_vt:.3f}  kg")

if __name__ == "__main__":
    from sizing import aileron, wing
    from sizing.wing import SizingInputs
    s = wing.run(SizingInputs())
    a = aileron.run(s)
    p = propulsion.run(s)
    summary(run(s, a, p, "airfoils/MH112.dat", "airfoils/NACA0010.dat"))