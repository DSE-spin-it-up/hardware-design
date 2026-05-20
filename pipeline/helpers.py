"""Small pipeline helpers: airfoil resolution, drag wrappers, LLT at CL, polar sweep."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerodynamics import drag_buildup
from aerodynamics.airfoil_polar import AirfoilPolar, get_airfoil_polar
from aerodynamics.drag_buildup import DragInputs, DragResult
from aerodynamics.llt import FlightCondition, LLTResult, WingGeometry, solve_llt
from aerodynamics.stability import (
    calculate_CM_wing,
    calculate_tail_loading,
    controllability_line_ShS,
    downwash_gradient,
    lift_slope_A_minus_h,
    lift_slope_from_llt,
    stability_line_ShS,
)
from propulsion.sizing import PropulsionResult
from sizing.aileron import AileronResult
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
    propulsion: PropulsionResult,
    wing_airfoil: str,
    tail_airfoil: str = "airfoils/NACA0010.dat",
) -> DragResult:
    rods = rods_mod.run(sizing, aileron, propulsion, wing_airfoil, tail_airfoil)   # ← pass airfoil_path
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
    cd_i_tail: float = 0.0,
) -> float:
    """Total CD at the operating point: CD0 buildup + wing induced + tail induced + payload.

    `cd_i_tail` must already be referenced to the wing area (S_w), not S_tail.
    """
    s = sizing.inputs
    CD_payload = s.Cd_payload * s.S_payload / (s.n_drones * sizing.Sw)
    return drag.CD0 + llt.CD_i + cd_i_tail + CD_payload


def tail_drag_at_cruise(
    sizing: SizingResult,
    polar: AirfoilPolar,
    llt: LLTResult,
    *,
    x_cg: float,
    tail_airfoil: str,
) -> dict:
    """Tail trim loading + induced drag at cruise.

    Returns the `calculate_tail_loading` dict with one extra key
    `CD_i_tail_wing_ref` = CD_i_tail · S_t / S_w, ready to add into the
    aircraft CD (which is referenced to S_w).
    """
    wing_geom = WingGeometry(b=sizing.inputs.b, S=sizing.Sw, taper=sizing.inputs.lam)
    tail_geom = WingGeometry(b=sizing.bh, S=sizing.Sh, taper=sizing.inputs.lam_t)

    Re_tail = sizing.rho * sizing.inputs.V_cruise * tail_geom.c_mean / 1.7894e-5
    tail_polar = get_airfoil_polar(
        tail_airfoil, Re=Re_tail, M=0.0,
        alpha_range=(-10.0, 10.0, 0.5), use_cache=True,
    )

    flight = FlightCondition(
        V_inf=sizing.inputs.V_cruise, rho=sizing.rho, CL_target=llt.CL,
    )

    result = calculate_tail_loading(
        wing_polar=polar,
        wing_geometry=wing_geom,
        tail_polar=tail_polar,
        tail_geometry=tail_geom,
        flight=flight,
        alpha_cruise=llt.alpha_root,
        CL_wing_cruise=llt.CL,
        l_tail=sizing.lh,
        x_cg=x_cg,
    )
    result["CD_i_tail_wing_ref"] = result["CD_i_tail"] * sizing.Sh / sizing.Sw
    return result


@dataclass
class ScissorData:
    """Scissor plot data (stability + controllability) + derived aero parameters."""
    # Sweep arrays
    x_cg: np.ndarray         # [m from LEMAC]
    ShS_stab: np.ndarray     # required S_h/S for static stability
    ShS_ctrl: np.ndarray     # required S_h/S for trim/controllability
    # Geometry
    b_f: float               # fuselage width [m]
    S_net: float             # exposed wing area [m²]
    r: float                 # 2·l_h / b_wing [-]
    # Derived aero — slopes (stability)
    CL_alpha_w: float        # [1/rad]
    CL_alpha_h: float        # [1/rad]
    CL_alpha_A_h: float      # [1/rad]
    dep_da: float            # dε/dα [-]
    # Derived aero — trim point CLs (controllability)
    CL_h: float              # tail CL at controllability condition
    CL_A_h: float            # aircraft-less-tail CL at controllability condition
    Cm_ac: float             # 3D-corrected wing CM about wing AC
    # Knobs
    SM: float
    Vh_V: float
    x_ac: float              # wing AC, [m from LEMAC]
    # Reference geometry
    c: float                 # MAC [m]
    l_h: float               # tail arm CG→tail AC [m]
    # Current operating point
    x_cg_current: float
    ShS_current: float


def compute_scissor_data(
    sizing: SizingResult,
    fus: FuselageResult,
    wing_llt: LLTResult,
    tail_llt: LLTResult,
    *,
    x_cg_current: float,
    y_cg: float = 0.0,
    SM: float = 0.05,
    Vh_V: float = 0.85,
    x_cg_range: tuple[float, float] | None = None,
    n_points: int = 200,
    CL_h: float | None = None,
    CL_A_h: float | None = None,
    Cm_ac: float | None = None,
) -> ScissorData:
    """Build the scissor (stability) line and bundle all derived parameters.

    Lift slopes are taken analytically from the existing wing/tail LLT solves
    (CL is exactly linear in α_root in classical LLT). Downwash dε/dα is
    Slingerland with Λ=0 and m_tv=0. x_ac is the wing MAC quarter chord.
    All x positions are measured from LEMAC.
    """
    s = sizing

    wing_geom = WingGeometry(b=s.inputs.b, S=s.Sw, taper=s.inputs.lam)

    # --- Net (exposed) wing area: subtract the trapezoidal slab inside the fuselage ---
    b_f = fus.width
    c_at_fus_edge = float(wing_geom.chord(np.array([b_f / 2.0]))[0])
    S_cov = b_f * (wing_geom.c_root + c_at_fus_edge) / 2.0
    S_net = s.Sw - S_cov

    # --- 3D lift-curve slopes from existing LLT solves ---
    CL_alpha_w = lift_slope_from_llt(wing_llt)
    CL_alpha_h = lift_slope_from_llt(tail_llt)
    CL_alpha_A_h = lift_slope_A_minus_h(CL_alpha_w, s.inputs.b, b_f, s.Sw, S_net)

    # --- Slingerland downwash (Λ=0, m_tv=0) ---
    r = 2.0 * s.lh / s.inputs.b
    dep_da = downwash_gradient(
        CL_alpha_w=CL_alpha_w, A_wing=s.inputs.AR, r=r, m_tv=0.0, sweep_qc=0.0,
    )

    # --- CG sweep + stability line ---
    if x_cg_range is None:
        x_cg_range = (0.0, s.c_root)
    x_cg = np.linspace(x_cg_range[0], x_cg_range[1], n_points)
    x_ac = 0.25 * s.c  # MAC quarter chord, from LEMAC

    ShS_stab = stability_line_ShS(
        x_cg, x_ac=x_ac, c=s.c, l_h=s.lh,
        CL_alpha_h=CL_alpha_h, CL_alpha_A_h=CL_alpha_A_h,
        dep_da=dep_da, Vh_V=Vh_V, SM=SM,
    )

    # --- Controllability line: uses trim-point CL values + wing Cm_ac ---
    # Defaults: cruise condition reuses existing LLT outputs. Override for
    # the conservative landing case.
    if CL_h is None:
        AR_tail = s.ARt
        CL_h = -0.35 * AR_tail ** (1.0 / 3.0)
    if CL_A_h is None:
        CL_A_h = wing_llt.CL  # wing-only proxy, ignoring fuselage lift contribution
    if Cm_ac is None:
        Cm_ac = calculate_CM_wing(wing_llt.polar, wing_llt.wing, wing_llt.alpha_root)
        payload_per_drone = sizing.inputs.m_payload / sizing.inputs.n_drones
        Cm_ac -= payload_per_drone * 9.81 * y_cg / (sizing.q_cruise * sizing.Sw * sizing.c)

    ShS_ctrl = controllability_line_ShS(
        x_cg, x_ac=x_ac, c=s.c, l_h=s.lh,
        CL_h=CL_h, CL_A_h=CL_A_h, Cm_ac=Cm_ac, Vh_V=Vh_V,
    )

    return ScissorData(
        x_cg=x_cg, ShS_stab=ShS_stab, ShS_ctrl=ShS_ctrl,
        b_f=b_f, S_net=S_net, r=r,
        CL_alpha_w=CL_alpha_w, CL_alpha_h=CL_alpha_h,
        CL_alpha_A_h=CL_alpha_A_h, dep_da=dep_da,
        CL_h=CL_h, CL_A_h=CL_A_h, Cm_ac=Cm_ac,
        SM=SM, Vh_V=Vh_V, x_ac=x_ac,
        c=s.c, l_h=s.lh,
        x_cg_current=x_cg_current,
        ShS_current=s.Sh / s.Sw,
    )


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