"""Zero-lift drag (CD0) buildup for wing + fuselage + tail + tail boom.

Uses Raymer-style component buildup:
    CD0 = sum_i (Cf_i * FF_i * Q_i * Swet_i) / S_ref

where Q_i is the per-component interference factor and S_ref = wing area.
Fuselage parameters (Swet, fineness, length) are natively ingested as 
ellipsoid values from the fuselage sizing module.
"""
from dataclasses import dataclass
import numpy as np

from aerodynamics.airfoil_geometry import AirfoilGeometry, airfoil_thickness_to_chord
from sizing.wing import R_air, SizingResult, gamma_air
from sizing.fuselage import FuselageResult
from structures.rods import RodResult


def _max_tc_with_location(airfoil: str) -> tuple[float, float]:
    """Return (max t/c, x/c at max thickness) for either a .dat path or NACA digits."""
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
    Q_wing: float = 1.0
    Q_v_tail: float = 1.03          # conventional aft tail
    Q_fus: float = 1.0
    Q_boom: float = 1.0
    Q_c_tail: float = 1.04


@dataclass
class DragResult:
    inputs: DragInputs
    Re_wing: float
    Re_tail_h: float
    Re_tail_v: float
    Re_fus: float
    Re_boom: float
    M_cruise: float
    M_tail: float
    Cf_wing: float
    FF_wing: float
    Swet_wing: float
    Cf_tail_h: float
    Cf_tail_v: float
    FF_tail_h: float
    FF_tail_v: float
    Swet_tail_h: float
    Swet_tail_v: float
    Cf_fus: float
    FF_fus: float
    Swet_fus: float
    Cf_boom: float
    FF_boom: float
    Swet_boom: float
    CD0_wing: float
    CD0_tail_h: float
    CD0_tail_v: float
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
    M_tail = rods.Vh_V * M_cruise
    V_tail = rods.Vh_V * V

    tc_w, xtc_w = _max_tc_with_location(i.wing_airfoil)
    tc_t, xtc_t = _max_tc_with_location(i.tail_airfoil)

    Re_wing = rho * V * s.c / MU_AIR
    Re_tail_h = rho * V_tail * s.ch / MU_AIR
    Re_tail_v = rho * V_tail * s.cv / MU_AIR
    # Uses outer ellipsoid aerodynamic length
    Re_fus  = rho * V * f.length / MU_AIR

    boom_diameter = rods.d_t
    boom_length   = s.L_boom
    Re_boom = rho * V * boom_length / MU_AIR

    Cf_wing = _skin_friction_turbulent(Re_wing, M_cruise)
    Cf_tail_h = _skin_friction_turbulent(Re_tail_h, M_cruise)
    Cf_tail_v = _skin_friction_turbulent(Re_tail_v, M_cruise)
    Cf_fus  = _skin_friction_turbulent(Re_fus,  M_cruise)
    Cf_boom = _skin_friction_turbulent(Re_boom, M_cruise)

    FF_wing = _form_factor_lifting(tc_w, xtc_w, M_cruise, i.sweep_wing)
    FF_tail_h = _form_factor_lifting(tc_t, xtc_t, M_tail, i.sweep_tail)
    FF_tail_v = _form_factor_lifting(tc_t, xtc_t, M_cruise, i.sweep_tail)
    # Uses equivalent ellipsoid fineness
    FF_fus  = _form_factor_fuselage(f.fineness)
    FF_boom = _form_factor_fuselage(boom_length / boom_diameter)

    Swet_wing = 2.0 * s.Sw
    Swet_tail_h = 2.0 * s.Sh
    Swet_tail_v = 2.0 * s.Sv
    # Uses exact Knud Thomsen ellipsoid Swet
    Swet_fus  = f.Swet
    Swet_boom = np.pi * boom_diameter * boom_length

    S_ref    = s.Sw
    CD0_wing = Cf_wing * FF_wing * i.Q_wing * Swet_wing / S_ref
    CD0_tail_h = Cf_tail_h * FF_tail_h * i.Q_c_tail * Swet_tail_h / S_ref
    CD0_tail_v = Cf_tail_v * FF_tail_v * i.Q_c_tail * Swet_tail_v / S_ref
    CD0_fus  = Cf_fus  * FF_fus  * i.Q_fus  * Swet_fus  / S_ref
    CD0_boom = Cf_boom * FF_boom * i.Q_boom * Swet_boom / S_ref
    CD0      = CD0_wing + CD0_tail_h + CD0_tail_v + CD0_fus + CD0_boom

    return DragResult(
        inputs=inputs,
        Re_wing=Re_wing, Re_tail_h=Re_tail_h, Re_tail_v=Re_tail_v, Re_fus=Re_fus, Re_boom=Re_boom,
        M_cruise=M_cruise, M_tail=M_tail,
        Cf_wing=Cf_wing, FF_wing=FF_wing, Swet_wing=Swet_wing,
        Cf_tail_h=Cf_tail_h, Cf_tail_v=Cf_tail_v, FF_tail_h=FF_tail_h, FF_tail_v=FF_tail_v, Swet_tail_h=Swet_tail_h,
        Swet_tail_v=Swet_tail_v, Cf_fus=Cf_fus,   FF_fus=FF_fus,   Swet_fus=Swet_fus,
        Cf_boom=Cf_boom, FF_boom=FF_boom, Swet_boom=Swet_boom,
        CD0_wing=CD0_wing, CD0_tail_h=CD0_tail_h, CD0_tail_v=CD0_tail_v, CD0_fus=CD0_fus,
        CD0_boom=CD0_boom, CD0=CD0,
    )


def summary(r: DragResult) -> None:
    print("\n--- Drag buildup (CD0) ---")
    print(f"  Re wing / hor. tail / ver. tail / fus / boom : {r.Re_wing:.2e} / {r.Re_tail_h:.2e} / {r.Re_tail_v:.2e} / {r.Re_fus:.2e} / {r.Re_boom:.2e}")
    print(f"  M cruise / tail             : {r.M_cruise:.4f} / {r.M_tail:.4f}")
    print(f"  Cf  wing / hor. tail / ver.tail / fus / boom: {r.Cf_wing:.5f} / {r.Cf_tail_h:.5f} / {r.Cf_tail_v:.5f} / {r.Cf_fus:.5f} / {r.Cf_boom:.5f}")
    print(f"  FF  wing / hor. tail / ver. tail / fus / boom: {r.FF_wing:.3f} / {r.FF_tail_h:.3f} / {r.FF_tail_v:.3f} / {r.FF_fus:.3f} / {r.FF_boom:.3f}")
    print(f"  Swet wing / hor. tail / ver. tail / fus / boom: {r.Swet_wing:.3f} / {r.Swet_tail_h:.3f} / {r.Swet_tail_v:.3f} / {r.Swet_fus:.3f} / {r.Swet_boom:.3f}  m²")

    total = r.CD0 if r.CD0 > 0 else 1.0
    pct = lambda x: 100 * x / total  
    print(f"  CD0 wing     : {r.CD0_wing:.5f}  ({pct(r.CD0_wing):5.1f}%)")
    print(f"  CD0 hor. tail: {r.CD0_tail_h:.5f}  ({pct(r.CD0_tail_h):5.1f}%)")
    print(f"  CD0 ver. tail: {r.CD0_tail_v:.5f}  ({pct(r.CD0_tail_v):5.1f}%)")
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