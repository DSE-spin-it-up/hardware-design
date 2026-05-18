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

Tail rod: cantilever with the tail download as a point load at the tip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerodynamics.airfoil_geometry import AirfoilGeometry
from sizing.aileron import AileronResult
from sizing.wing import SizingResult
from structures.materials import CFRP


@dataclass
class RodInputs:
    material: CFRP = field(default_factory=CFRP)
    safety_factor: float = 1.2
    defl_max: float = 0.05            # [m] max tip deflection
    d_to_section_ratio: float = 0.8   # rod OD as a fraction of local section thickness
    CLt_max: float = 1.0              # tail max lift coefficient for tail-rod sizing
    tail_tc: float = 0.10             # tail-airfoil t/c for the geometric fit


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
    t_aileron: float
    mass_aileron: float
    defl_aileron: float
    fail_mode_aileron: str
    # Tail rod
    d_t: float
    t_t: float
    mass_t: float
    defl_t: float
    fail_mode_t: str

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


def _t_for_defl_half_cantilever_udl(
    F_total: float, L_span: float, E: float, d: float, defl_max: float
) -> float:
    """Wall thickness for tip deflection ≤ defl_max (half-cantilever, UDL).

    δ = F·L³/(8·E·I); I = π·t·d³/8  →  t = F·L³/(π·E·d³·δ)
    """
    F, L = F_total / 2, L_span / 2
    return F * L ** 3 / (np.pi * E * d ** 3 * defl_max)


def _t_for_defl_cantilever_point(
    F: float, L: float, E: float, d: float, defl_max: float
) -> float:
    """Wall thickness for tip deflection ≤ defl_max (cantilever, point load).

    δ = F·L³/(3·E·I); I = π·t·d³/8  →  t = 8·F·L³/(3·π·E·d³·δ)
    """
    return 8 * F * L ** 3 / (3 * np.pi * E * d ** 3 * defl_max)


def _t_for_stress(M: float, sigma_lim: float, d: float) -> float:
    """Wall thickness so bending stress ≤ sigma_lim at outer fibre."""
    return 4 * M / (np.pi * sigma_lim * d ** 2)


def _defl_half_cantilever_udl(
    F_total: float, L_span: float, E: float, I: float
) -> float:
    F, L = F_total / 2, L_span / 2
    return F * L ** 3 / (8 * E * I)


def _defl_cantilever_point(F: float, L: float, E: float, I: float) -> float:
    return F * L ** 3 / (3 * E * I)


def _tube_mass(length: float, d: float, t: float, rho: float) -> float:
    return ((d / 2) ** 2 - (d / 2 - t) ** 2) * np.pi * length * rho


def _check_wall(t: float, d: float, label: str) -> None:
    if t >= d / 2:
        raise ValueError(
            f"{label}: required wall thickness {t * 1000:.2f} mm exceeds rod radius "
            f"{d / 2 * 1000:.2f} mm — rod cannot satisfy criteria at this diameter."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    sizing: SizingResult,
    aileron: AileronResult,
    airfoil_path: str,                 # ← required, no default
    inputs: RodInputs | None = None,
) -> RodResult:
    if inputs is None:
        inputs = RodInputs()
    i = inputs
    s = sizing
    si = s.inputs
    mat = i.material
    sigma_lim = mat.s_c / i.safety_factor
    E = mat.E
    rho_mat = mat.rho

    # Load airfoil geometry once for local thickness queries.
    airfoil = AirfoilGeometry(airfoil_path)

    # Total wing lift and span (shared by both wing rods).
    L_lift = s.CL * s.q_cruise * s.Sw    # [N]
    b_w = si.b

    # ------------------------------------------------------------------ #
    # Spar rod — at max-thickness x/c                                     #
    # ------------------------------------------------------------------ #
    tc_spar, _ = airfoil.compute_maximum_thickness()   # (t/c, x/c)
    section_h_spar = s.c_root * tc_spar
    d_spar = i.d_to_section_ratio * section_h_spar

    M_spar = L_lift * b_w / 8             # half-cantilever UDL max bending moment

    t_spar_defl = _t_for_defl_half_cantilever_udl(L_lift, b_w, E, d_spar, i.defl_max)
    t_spar_comp = _t_for_stress(M_spar, sigma_lim, d_spar)
    if t_spar_defl >= t_spar_comp:
        t_spar, fail_spar = t_spar_defl, "deflection"
    else:
        t_spar, fail_spar = t_spar_comp, "compressive"
    _check_wall(t_spar, d_spar, "Spar rod")
    defl_spar = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_spar, d_spar))
    mass_spar = _tube_mass(b_w, d_spar, t_spar, rho_mat)

    # ------------------------------------------------------------------ #
    # Aileron rod — at hinge x/c = 1 - c_aileron/c_wing                  #
    # ------------------------------------------------------------------ #
    x_hinge = 1.0 - aileron.inputs.c_aileron_to_c_wing
    tc_aileron, _, _ = airfoil.compute_thickness(x_hinge)   # local t/c at hinge
    section_h_aileron = s.c_root * tc_aileron
    d_aileron = i.d_to_section_ratio * section_h_aileron

    # Aileron rod carries only the aileron hinge load — modelled as a
    # simple beam with the same UDL as the spar (conservative; actual
    # hinge loads are lower).  Primarily deflection-governed at this
    # smaller diameter.
    t_ail_defl = _t_for_defl_half_cantilever_udl(L_lift, b_w, E, d_aileron, i.defl_max)
    t_ail_comp = _t_for_stress(M_spar, sigma_lim, d_aileron)
    if t_ail_defl >= t_ail_comp:
        t_aileron, fail_aileron = t_ail_defl, "deflection"
    else:
        t_aileron, fail_aileron = t_ail_comp, "compressive"
    _check_wall(t_aileron, d_aileron, "Aileron rod")
    defl_aileron = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_aileron, d_aileron))
    mass_aileron = _tube_mass(b_w, d_aileron, t_aileron, rho_mat)

    # ------------------------------------------------------------------ #
    # Tail rod                                                             #
    # ------------------------------------------------------------------ #
    section_thickness_t = s.ct * i.tail_tc
    d_t = i.d_to_section_ratio * section_thickness_t
    F_tail = s.Sh * i.CLt_max * s.q_cruise   # tail download as point load [N]
    L_t = s.L_tail
    M_t = F_tail * L_t

    t_t_defl = _t_for_defl_cantilever_point(F_tail, L_t, E, d_t, i.defl_max)
    t_t_comp = _t_for_stress(M_t, sigma_lim, d_t)
    if t_t_defl >= t_t_comp:
        t_t, fail_t = t_t_defl, "deflection"
    else:
        t_t, fail_t = t_t_comp, "compressive"
    _check_wall(t_t, d_t, "Tail rod")
    defl_t = _defl_cantilever_point(F_tail, L_t, E, _I_tube(t_t, d_t))
    mass_t = _tube_mass(L_t, d_t, t_t, rho_mat)

    return RodResult(
        inputs=inputs,
        d_spar=d_spar, t_spar=t_spar, mass_spar=mass_spar,
        defl_spar=defl_spar, fail_mode_spar=fail_spar,
        d_aileron=d_aileron, t_aileron=t_aileron, mass_aileron=mass_aileron,
        defl_aileron=defl_aileron, fail_mode_aileron=fail_aileron,
        d_t=d_t, t_t=t_t, mass_t=mass_t, defl_t=defl_t, fail_mode_t=fail_t,
    )


def summary(r: RodResult) -> None:
    print("\n--- Spar rod (one of two) ---")
    print(f"  Outer diameter       : {r.d_spar * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_spar * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_spar * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_spar}")
    print(f"  Mass (each)          : {r.mass_spar:.3f}  kg")

    print("\n--- Aileron rod (one of two) ---")
    print(f"  Outer diameter       : {r.d_aileron * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_aileron * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_aileron * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_aileron}")
    print(f"  Mass (each)          : {r.mass_aileron:.3f}  kg")

    print("\n--- Tail rod ---")
    print(f"  Outer diameter       : {r.d_t * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_t * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_t * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_t}")
    print(f"  Mass                 : {r.mass_t:.3f}  kg")


if __name__ == "__main__":
    from sizing import aileron, wing
    from sizing.wing import SizingInputs
    s = wing.run(SizingInputs())
    a = aileron.run(s)
    summary(run(s, a, "airfoils/MH112.dat"))