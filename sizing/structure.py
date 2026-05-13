"""Rod sizing for the wing and tail.

Both rods are sized closed-form for the thinnest CFRP wall that
simultaneously satisfies a tip-deflection limit and a compressive-stress
limit. The outer diameter is fixed by geometry — the rod must fit inside
the local section thickness.

Wing rod: half-cantilever with the wing UDL.
Tail rod: cantilever with the tail download as a point load at the tip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sizing.initial_sizing import SizingResult
from sizing.materials import CFRP


@dataclass
class StructureInputs:
    material: CFRP = field(default_factory=CFRP)
    safety_factor: float = 1.2
    defl_max: float = 0.05            # [m] max tip deflection
    d_to_section_ratio: float = 0.8   # rod OD as a fraction of local section thickness
    CLt_max: float = 1.0              # tail max lift coefficient for tail-rod sizing
    tail_tc: float = 0.10             # tail-airfoil t/c for the geometric fit


@dataclass
class StructureResult:
    inputs: StructureInputs
    # Wing rod (one rod; total wing mass uses 2× this)
    d_w: float
    t_w: float
    mass_w: float
    defl_w: float
    fail_mode_w: str   # "deflection" | "compressive"
    # Tail rod
    d_t: float
    t_t: float
    mass_t: float
    defl_t: float
    fail_mode_t: str


def _I_tube(t: float, d: float) -> float:
    """Thin-wall tube second moment of area: I = π·t·d³/8."""
    return np.pi * t * d ** 3 / 8


def _t_for_defl_half_cantilever_udl(F_total: float, L_span: float, E: float, d: float, defl_max: float) -> float:
    """Wall thickness needed so tip deflection of a half-cantilever (UDL) ≤ defl_max.

    Half-wing model: each half is a cantilever of length L_span/2 carrying
    half the total load (F_total/2) distributed uniformly.
    δ = F·L³/(8·E·I); with I = π·t·d³/8 ⇒ t = F·L³/(π·E·d³·δ).
    """
    F, L = F_total / 2, L_span / 2
    return F * L ** 3 / (np.pi * E * d ** 3 * defl_max)


def _t_for_defl_cantilever_point(F: float, L: float, E: float, d: float, defl_max: float) -> float:
    """Wall thickness needed so tip deflection of a cantilever (point load) ≤ defl_max.

    δ = F·L³/(3·E·I); with I = π·t·d³/8 ⇒ t = 8·F·L³/(3·π·E·d³·δ).
    """
    return 8 * F * L ** 3 / (3 * np.pi * E * d ** 3 * defl_max)


def _t_for_stress(M: float, sigma_lim: float, d: float) -> float:
    """Wall thickness needed so bending stress ≤ sigma_lim at outer fibre."""
    return 4 * M / (np.pi * sigma_lim * d ** 2)


def _defl_half_cantilever_udl(F_total: float, L_span: float, E: float, I: float) -> float:
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


def run(sizing: SizingResult, inputs: StructureInputs | None = None) -> StructureResult:
    if inputs is None:
        inputs = StructureInputs()
    i = inputs
    s = sizing
    si = s.inputs
    mat = i.material
    sigma_lim = mat.s_c / i.safety_factor
    E = mat.E
    rho = mat.rho

    # ----- Wing rod -----
    section_thickness_w = s.c * s.t_over_c_root
    d_w = i.d_to_section_ratio * section_thickness_w
    L_lift = s.CL * s.q_cruise * s.Sw    # total wing lift [N]
    b_w = si.b
    M_w = L_lift * b_w / 8               # half-cantilever UDL max bending moment

    t_w_defl = _t_for_defl_half_cantilever_udl(L_lift, b_w, E, d_w, i.defl_max)
    t_w_comp = _t_for_stress(M_w, sigma_lim, d_w)
    if t_w_defl >= t_w_comp:
        t_w, fail_w = t_w_defl, "deflection"
    else:
        t_w, fail_w = t_w_comp, "compressive"
    _check_wall(t_w, d_w, "Wing rod")
    defl_w = _defl_half_cantilever_udl(L_lift, b_w, E, _I_tube(t_w, d_w))
    mass_w = _tube_mass(b_w, d_w, t_w, rho)

    # ----- Tail rod -----
    section_thickness_t = s.ct * i.tail_tc
    d_t = i.d_to_section_ratio * section_thickness_t
    F_tail = s.Sh * i.CLt_max * s.q_cruise   # tail download as a point load [N]
    L_t = s.L_tail
    M_t = F_tail * L_t                       # cantilever with point load at the tip

    t_t_defl = _t_for_defl_cantilever_point(F_tail, L_t, E, d_t, i.defl_max)
    t_t_comp = _t_for_stress(M_t, sigma_lim, d_t)
    if t_t_defl >= t_t_comp:
        t_t, fail_t = t_t_defl, "deflection"
    else:
        t_t, fail_t = t_t_comp, "compressive"
    _check_wall(t_t, d_t, "Tail rod")
    defl_t = _defl_cantilever_point(F_tail, L_t, E, _I_tube(t_t, d_t))
    mass_t = _tube_mass(L_t, d_t, t_t, rho)

    return StructureResult(
        inputs=inputs,
        d_w=d_w, t_w=t_w, mass_w=mass_w, defl_w=defl_w, fail_mode_w=fail_w,
        d_t=d_t, t_t=t_t, mass_t=mass_t, defl_t=defl_t, fail_mode_t=fail_t,
    )


def summary(r: StructureResult) -> None:
    print("\n--- Wing rod (one of two) ---")
    print(f"  Outer diameter       : {r.d_w * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_w * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_w * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_w}")
    print(f"  Mass (each)          : {r.mass_w:.3f}  kg")

    print("\n--- Tail rod ---")
    print(f"  Outer diameter       : {r.d_t * 1000:.2f}  mm")
    print(f"  Wall thickness       : {r.t_t * 1000:.2f}  mm")
    print(f"  Tip deflection       : {r.defl_t * 1000:.2f}  mm")
    print(f"  Sizing criterion     : {r.fail_mode_t}")
    print(f"  Mass                 : {r.mass_t:.3f}  kg")


if __name__ == "__main__":
    from sizing import initial_sizing
    from sizing.initial_sizing import SizingInputs
    summary(run(initial_sizing.run(SizingInputs())))
