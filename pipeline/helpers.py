"""Small pipeline helpers: airfoil resolution, drag wrappers, LLT at CL, polar sweep."""
from __future__ import annotations

import numpy as np

from aerodynamics import drag_buildup
from sizing.aileron import AileronResult
from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.drag_buildup import DragInputs, DragResult
from aerodynamics.llt import FlightCondition, LLTResult, WingGeometry, solve_llt
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from structures import rods as rods_mod


def resolve_airfoil(airfoil: str | None) -> str:
    if airfoil is not None:
        return airfoil
    return input("Enter airfoil (.dat path or NACA digits): ").strip()


def estimate_cd0(
    sizing: SizingResult,
    fus: FuselageResult,
    aileron: AileronResult,
    wing_airfoil: str,
    tail_airfoil: str = "airfoils/NACA0010.dat",
) -> DragResult:
    rods = rods_mod.run(sizing, aileron, wing_airfoil)   # ← pass airfoil_path
    return drag_buildup.run(
        sizing,
        fus,
        rods,
        DragInputs(wing_airfoil=wing_airfoil, tail_airfoil=tail_airfoil),
    )


def full_drag_estimate(
    sizing: SizingResult,
    drag: DragResult,
    llt: LLTResult,
) -> float:
    """Total CD at the operating point: CD0 buildup + induced + payload bluff-body."""
    s = sizing.inputs
    CD_payload = s.Cd_payload * s.S_payload / (s.n_drones * sizing.Sw)
    return drag.CD0 + llt.CD_i + CD_payload


def llt_at_cl(
    sizing: SizingResult,
    polar: AirfoilPolar,
    CL_target: float,
) -> LLTResult:
    wing = WingGeometry(b=sizing.inputs.b, S=sizing.Sw, taper=sizing.inputs.lam)
    flight = FlightCondition(
        V_inf=sizing.inputs.V_cruise,
        rho=sizing.rho,
        CL_target=CL_target,
    )
    return solve_llt(wing, polar, flight, N=50)


def wing_drag_polar(
    sizing: SizingResult,
    polar: AirfoilPolar,
    alpha_deg_range: tuple[float, float, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep α and build (CL, CD) at the wing level."""
    wing = WingGeometry(b=sizing.inputs.b, S=sizing.Sw, taper=sizing.inputs.lam)
    a_lo, a_hi, n = alpha_deg_range
    alphas = np.radians(np.linspace(a_lo, a_hi, n))
    CL_arr = np.empty(n)
    CD_arr = np.empty(n)
    for k, a in enumerate(alphas):
        flight = FlightCondition(V_inf=sizing.inputs.V_cruise, rho=sizing.rho, alpha_root=a)
        res = solve_llt(wing, polar, flight, N=30)
        # Span-averaged section Cd from the airfoil polar at the local Cl.
        Cd_profile = float(np.mean(polar.Cd_p(res.Cl_local)))
        CL_arr[k] = res.CL
        CD_arr[k] = Cd_profile + res.CD_i
    return CL_arr, CD_arr