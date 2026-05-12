"""Zero-lift drag (CD0) buildup for wing + fuselage + tail.

Uses Raymer-style component buildup:
    CD0 = sum_i (Cf_i * FF_i * Q_i * Swet_i) / S_ref

with Q_i = 1.0 (interference factor) and S_ref = wing area.
"""
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_shape import AirfoilGeometry
from sizing.initial_sizing import R_air, SizingResult, gamma_air
from sizing.fuselage import FuselageResult


# Air dynamic viscosity at standard cruise temperature [Pa·s].
MU_AIR = 1.82e-5


@dataclass
class DragInputs:
    wing_airfoil: str = "airfoils/MH112.dat"
    tail_airfoil: str = "airfoils/NACA0010.dat"
    sweep_wing: float = 0.0       # quarter-chord sweep [rad]
    sweep_tail: float = 0.0       # [rad]
    Vh_V: float = 0.85            # tail-to-wing dynamic-pressure ratio (V_tail/V_cruise)


@dataclass
class DragResult:
    inputs: DragInputs
    # Reynolds & Mach
    Re_wing: float
    Re_tail: float
    Re_fus: float
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
    # Aggregate
    CD0_wing: float
    CD0_tail: float
    CD0_fus: float
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
    wing_af = AirfoilGeometry(i.wing_airfoil)
    tc_w, xtc_w = wing_af.compute_maximum_thickness()
    tail_af = AirfoilGeometry(i.tail_airfoil)
    tc_t, xtc_t = tail_af.compute_maximum_thickness()

    # Reynolds numbers
    Re_wing = rho * V * s.c / MU_AIR
    Re_tail = rho * V_tail * s.ct / MU_AIR
    Re_fus = rho * V * f.length / MU_AIR

    # Skin friction
    Cf_wing = _skin_friction_turbulent(Re_wing, M_cruise)
    Cf_tail = _skin_friction_turbulent(Re_tail, M_cruise)
    Cf_fus = _skin_friction_turbulent(Re_fus, M_cruise)

    # Form factors
    FF_wing = _form_factor_lifting(tc_w, xtc_w, M_cruise, i.sweep_wing)
    FF_tail = _form_factor_lifting(tc_t, xtc_t, M_tail, i.sweep_tail)
    FF_fus = _form_factor_fuselage(f.fineness)

    # Wetted areas (lifting surfaces ≈ 2 × planform).
    Swet_wing = 2.0 * s.Sw
    Swet_tail = 2.0 * s.St
    Swet_fus = f.Swet

    # Per-component CD0 referenced to wing area.
    S_ref = s.Sw
    CD0_wing = Cf_wing * FF_wing * Swet_wing / S_ref
    CD0_tail = Cf_tail * FF_tail * Swet_tail / S_ref * (i.Vh_V ** 2)
    CD0_fus = Cf_fus * FF_fus * Swet_fus / S_ref
    CD0 = CD0_wing + CD0_tail + CD0_fus

    return DragResult(
        inputs=inputs,
        Re_wing=Re_wing, Re_tail=Re_tail, Re_fus=Re_fus,
        M_cruise=M_cruise, M_tail=M_tail,
        Cf_wing=Cf_wing, FF_wing=FF_wing, Swet_wing=Swet_wing,
        Cf_tail=Cf_tail, FF_tail=FF_tail, Swet_tail=Swet_tail,
        Cf_fus=Cf_fus, FF_fus=FF_fus, Swet_fus=Swet_fus,
        CD0_wing=CD0_wing, CD0_tail=CD0_tail, CD0_fus=CD0_fus,
        CD0=CD0,
    )


def summary(r: DragResult) -> None:
    print("\n--- Drag buildup (CD0) ---")
    print(f"  Re wing / tail / fus : {r.Re_wing:.2e} / {r.Re_tail:.2e} / {r.Re_fus:.2e}")
    print(f"  M cruise / tail      : {r.M_cruise:.4f} / {r.M_tail:.4f}")
    print(f"  Cf wing/tail/fus     : {r.Cf_wing:.5f} / {r.Cf_tail:.5f} / {r.Cf_fus:.5f}")
    print(f"  FF wing/tail/fus     : {r.FF_wing:.3f} / {r.FF_tail:.3f} / {r.FF_fus:.3f}")
    print(f"  Swet wing/tail/fus   : {r.Swet_wing:.3f} / {r.Swet_tail:.3f} / {r.Swet_fus:.3f}  m²")
    print(f"  CD0 wing             : {r.CD0_wing:.5f}")
    print(f"  CD0 tail             : {r.CD0_tail:.5f}")
    print(f"  CD0 fuselage         : {r.CD0_fus:.5f}")
    print(f"  CD0 total            : {r.CD0:.5f}")


if __name__ == "__main__":
    from sizing import initial_sizing, fuselage as fus_mod
    s = initial_sizing._default
    f = fus_mod._default
    r = run(s, f)
    summary(r)
