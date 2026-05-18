"""Zero-lift drag (CD0) buildup for wing + fuselage + tail + tail boom.

Uses Raymer-style component buildup:
    CD0 = sum_i (Cf_i * FF_i * Q_i * Swet_i) / S_ref

where Q_i is the per-component interference factor and S_ref = wing area.
Tail-boom geometry (diameter, length) is taken directly from structures.rods.RodResult
so there is no duplication of sizing data.
"""
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_geometry import AirfoilGeometry, airfoil_thickness_to_chord
from sizing.wing import R_air, SizingResult, gamma_air
from sizing.fuselage import FuselageResult
from structures.rods import RodResult


def _max_tc_with_location(airfoil: str) -> tuple[float, float]:
    """Return (max t/c, x/c at max thickness) for either a .dat path or NACA digits.

    The 0.30 below is exact for NACA 4-/5-digit airfoils: both series share
    the same thickness polynomial, which peaks at x/c = 0.30. It is NOT a
    generic fallback — NACA 6-series (and other modern laminar-flow sections)
    have max thickness at 0.40-0.50 and would be silently mis-sized here.
    Pass a .dat file for anything outside the 4-/5-digit family.
    """
    if str(airfoil).endswith(".dat"):
        return AirfoilGeometry(airfoil).compute_maximum_thickness()
    return airfoil_thickness_to_chord(airfoil), 0.30


# Air dynamic viscosity at standard cruise temperature [Pa·s].
MU_AIR = 1.82e-5


@dataclass
class DragInputs:
    wing_airfoil: str = "airfoils/MH112.dat"
    tail_airfoil: str = "airfoils/NACA0010.dat"
    sweep_wing: float = 0.0       # quarter-chord sweep [rad]
    sweep_tail: float = 0.0       # [rad]
    Vh_V: float = 0.85            # V_tail/V_cruise — used for tail Re/Mach only
    # Raymer interference factors (Q_i in CD0 buildup)
    Q_wing: float = 1.0
    Q_tail: float = 1.03          # conventional aft tail
    Q_fus: float = 1.0
    Q_boom: float = 1.0         


@dataclass
class DragResult:
    inputs: DragInputs
    # Reynolds & Mach
    Re_wing: float
    Re_tail: float
    Re_fus: float
    Re_boom: float
    M_cruise: float
    M_tail: float
    # Per-component contributions
    Cf_wing: float
    FF_wing: float
    Swet_wing: float
    Cf_tail: float
    FF_tail: float
    Swet_tail: float
    Cf_fus: float
    FF_fus: float
    Swet_fus: float
    Cf_boom: float
    FF_boom: float
    Swet_boom: float
    # Aggregate
    CD0_wing: float
    CD0_tail: float
    CD0_fus: float
    CD0_boom: float
    CD0: float


def _skin_friction_turbulent(Re: float, M: float) -> float:
    """Raymer turbulent flat-plate skin friction with compressibility."""
    return 0.455 / (np.log10(Re) ** 2.58 * (1 + 0.144 * M ** 2) ** 0.65)


def _form_factor_lifting(max_tc: float, max_tc_loc: float, M: float, sweep: float) -> float:
    """Raymer form factor for wing/tail-like lifting surfaces."""
    return (1 + 0.6 / max_tc_loc * max_tc + 100 * max_tc ** 4) * (
        1.34 * M ** 0.18 * (np.cos(sweep)) ** 0.28
    )


def _form_factor_fuselage(fineness: float) -> float:
    """Raymer form factor for a fuselage-like body of revolution."""
    return 1 + 60 / fineness ** 3 + fineness / 400


def run(
    sizing: SizingResult,
    fuselage: FuselageResult,
    rods: RodResult,
    inputs: DragInputs | None = None,
) -> DragResult:
    if inputs is None:
        inputs = DragInputs()
    i = inputs
    s = sizing
    f = fuselage

    V = s.inputs.V_cruise
    rho = s.rho
    M_cruise = V / np.sqrt(gamma_air * R_air * s.T_isa)
    M_tail = i.Vh_V * M_cruise
    V_tail = i.Vh_V * V

    # Airfoil thicknesses
    tc_w, xtc_w = _max_tc_with_location(i.wing_airfoil)
    tc_t, xtc_t = _max_tc_with_location(i.tail_airfoil)

    # Reynolds numbers
    Re_wing = rho * V * s.c / MU_AIR
    Re_tail = rho * V_tail * s.ct / MU_AIR
    Re_fus  = rho * V * f.length / MU_AIR

    # Tail boom geometry from structural sizing — no duplication of data.
    # rods.d_t is the outer diameter [m]; s.L_tail is the cantilever length
    # used in rods.py, so both modules share the same source of truth.
    boom_diameter = rods.d_t
    boom_length   = s.L_tail
    Re_boom = rho * V * boom_length / MU_AIR

    # Skin friction
    Cf_wing = _skin_friction_turbulent(Re_wing, M_cruise)
    Cf_tail = _skin_friction_turbulent(Re_tail, M_cruise)
    Cf_fus  = _skin_friction_turbulent(Re_fus,  M_cruise)
    Cf_boom = _skin_friction_turbulent(Re_boom, M_cruise)

    # Form factors
    FF_wing = _form_factor_lifting(tc_w, xtc_w, M_cruise, i.sweep_wing)
    FF_tail = _form_factor_lifting(tc_t, xtc_t, M_tail,   i.sweep_tail)
    FF_fus  = _form_factor_fuselage(f.fineness)
    FF_boom = _form_factor_fuselage(boom_length / boom_diameter)

    # Wetted areas
    Swet_wing = 2.0 * s.Sw                          # lifting surface ≈ 2 × planform
    Swet_tail = 2.0 * s.St
    Swet_fus  = f.Swet
    Swet_boom = np.pi * boom_diameter * boom_length  # lateral surface of cylinder

    # Per-component CD0 referenced to wing area.
    S_ref    = s.Sw
    CD0_wing = Cf_wing * FF_wing * i.Q_wing * Swet_wing / S_ref
    CD0_tail = Cf_tail * FF_tail * i.Q_tail * Swet_tail / S_ref
    CD0_fus  = Cf_fus  * FF_fus  * i.Q_fus  * Swet_fus  / S_ref
    CD0_boom = Cf_boom * FF_boom * i.Q_boom * Swet_boom / S_ref
    CD0      = CD0_wing + CD0_tail + CD0_fus + CD0_boom

    return DragResult(
        inputs=inputs,
        Re_wing=Re_wing, Re_tail=Re_tail, Re_fus=Re_fus, Re_boom=Re_boom,
        M_cruise=M_cruise, M_tail=M_tail,
        Cf_wing=Cf_wing, FF_wing=FF_wing, Swet_wing=Swet_wing,
        Cf_tail=Cf_tail, FF_tail=FF_tail, Swet_tail=Swet_tail,
        Cf_fus=Cf_fus,   FF_fus=FF_fus,   Swet_fus=Swet_fus,
        Cf_boom=Cf_boom, FF_boom=FF_boom, Swet_boom=Swet_boom,
        CD0_wing=CD0_wing, CD0_tail=CD0_tail, CD0_fus=CD0_fus,
        CD0_boom=CD0_boom, CD0=CD0,
    )


def summary(r: DragResult) -> None:
    print("\n--- Drag buildup (CD0) ---")
    print(f"  Re wing / tail / fus / boom : {r.Re_wing:.2e} / {r.Re_tail:.2e} / {r.Re_fus:.2e} / {r.Re_boom:.2e}")
    print(f"  M cruise / tail             : {r.M_cruise:.4f} / {r.M_tail:.4f}")
    print(f"  Cf  wing / tail / fus / boom: {r.Cf_wing:.5f} / {r.Cf_tail:.5f} / {r.Cf_fus:.5f} / {r.Cf_boom:.5f}")
    print(f"  FF  wing / tail / fus / boom: {r.FF_wing:.3f} / {r.FF_tail:.3f} / {r.FF_fus:.3f} / {r.FF_boom:.3f}")
    print(f"  Swet wing / tail / fus / boom: {r.Swet_wing:.3f} / {r.Swet_tail:.3f} / {r.Swet_fus:.3f} / {r.Swet_boom:.3f}  m²")

    total = r.CD0 if r.CD0 > 0 else 1.0
    pct = lambda x: 100 * x / total  # noqa: E731
    print(f"  CD0 wing     : {r.CD0_wing:.5f}  ({pct(r.CD0_wing):5.1f}%)")
    print(f"  CD0 tail     : {r.CD0_tail:.5f}  ({pct(r.CD0_tail):5.1f}%)")
    print(f"  CD0 fuselage : {r.CD0_fus:.5f}  ({pct(r.CD0_fus):5.1f}%)")
    print(f"  CD0 boom     : {r.CD0_boom:.5f}  ({pct(r.CD0_boom):5.1f}%)")
    print(f"  CD0 total    : {r.CD0:.5f}")


if __name__ == "__main__":
    from sizing import aileron, fuselage as fus_mod, wing
    from sizing.wing import SizingInputs
    from structures import rods as rods_mod

    wing_airfoil = "airfoils/MH112.dat"
    s = wing.run(SizingInputs())
    f = fus_mod.run(s)
    a = aileron.run(s)
    r = rods_mod.run(s, a, wing_airfoil)
    summary(run(s, f, r, DragInputs(wing_airfoil=wing_airfoil)))