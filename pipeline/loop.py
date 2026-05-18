"""Sizing↔propulsion↔fuselage↔drag↔mass↔wing-area convergence loop.

`run_pipeline(config)` reads design knobs from a config module (any module
exposing the module-level constants in `pipeline.config`) and returns a
`PipelineResult` bundling every value the report / plot consumers need.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_geometry import airfoil_thickness_to_chord
from aerodynamics.airfoil_polar import AirfoilPolar, get_airfoil_polar
from aerodynamics.drag_buildup import DragResult
from aerodynamics.llt import LLTResult
from pipeline.helpers import (
    ScissorData,
    compute_scissor_data,
    estimate_cd0,
    full_drag_estimate,
    llt_at_cl,
    resolve_airfoil,
    tail_drag_at_cruise,
    wing_drag_polar,
)
from propulsion import sizing as prop_sizing
from propulsion.sizing import PropulsionResult
from sizing import aileron, fuselage, wing
from sizing.aileron import AileronResult
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from structures import rods
from structures.rods import RodResult
from weights import mass as weights_mass


@dataclass
class PipelineResult:
    # ----- Setup -----
    airfoil: str
    tail_airfoil: str
    tc: float
    polar: AirfoilPolar
    cl_max: float

    # ----- Converged design state -----
    sizing: SizingResult
    propulsion: PropulsionResult
    fus: FuselageResult
    drag: DragResult
    struct: RodResult
    control_surface: AileronResult
    masses: dict[str, float]
    cg: dict[str, float]

    # ----- Convergence history (for plot_convergence) -----
    cd0_history: list[float]
    mass_history: list[float]
    sw_history: list[float]
    cg_history: list[float]
    cd0_guess: float
    m_drone_guess: float
    sw_guess: float

    # ----- Post-loop analysis -----
    llt: LLTResult
    cl_req: float
    max_cl_local: float
    lift_achievable: bool
    cl_sweep: np.ndarray
    cd_drone_sweep: np.ndarray
    cd_full_sweep: np.ndarray
    cd_full_buildup: float
    cd_payload: float
    cd_i_tail: float       # tail induced drag, wing-area reference
    tail_loading: dict     # CL_tail, CD_i_tail (tail ref), e_tail, AR_tail
    scissor: ScissorData   # stability-line scissor plot data


@dataclass
class _DesignPass:
    """All sub-solver outputs for a given `sizing`."""
    propulsion: PropulsionResult
    fus: FuselageResult
    drag: DragResult
    struct: RodResult
    control_surface: AileronResult
    masses: dict[str, float]
    cg: dict[str, float]


def _run_design_pass(
    sizing: SizingResult,
    *,
    config,
    polar: AirfoilPolar,
    airfoil: str,
) -> _DesignPass:
    propulsion = prop_sizing.run(sizing, config.PROPULSION)
    fus = fuselage.run(
        sizing,
        config.FUSELAGE,
        battery_volume=propulsion.battery_volume,
        airfoil_path=airfoil,
    )
    # Aileron must be computed before rods and drag — both need the hinge x/c.
    control_surface = aileron.run(sizing, config.CONTROL_SURFACE, polar=polar)
    drag = estimate_cd0(
        sizing, fus, control_surface,
        wing_airfoil=airfoil,
        tail_airfoil=config.TAIL_AIRFOIL,
    )
    struct = rods.run(sizing, control_surface, airfoil, config.STRUCTURE)
    masses = weights_mass.total_mass(
        sizing=sizing,
        propulsion=propulsion,
        structure=struct,
        fus=fus,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
        materials=config.MATERIALS,
    )
    cg = weights_mass.compute_cg(
        sizing=sizing,
        propulsion=propulsion,
        structure=struct,
        fus=fus,
        aileron=control_surface,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
        materials=config.MATERIALS,
    )
    return _DesignPass(propulsion, fus, drag, struct, control_surface, masses, cg)


def _sw_closure_b(
    sizing: SizingResult,
    drag: DragResult,
    polar: AirfoilPolar,
    *,
    m_drone_eff: float,
    alpha_range: tuple[float, float, int],
) -> tuple[float, float]:
    """Pick the span that places the operating CL at max drone+payload L/D.

    Returns (b_new, CL_target).
    """
    s = sizing.inputs
    CL_sw, CD_wing_sw = wing_drag_polar(sizing, polar, alpha_range)
    CD_nonwing = drag.CD0_tail + drag.CD0_fus
    CD_payload = s.Cd_payload * s.S_payload / (s.n_drones * sizing.Sw)
    LD = CL_sw / (CD_wing_sw + CD_nonwing + CD_payload)
    CL_target = float(CL_sw[int(np.argmax(LD))])

    W = (m_drone_eff + s.m_payload / s.n_drones) * 9.80665
    Sw_new = W / (sizing.q_cruise * CL_target)
    b_new = float(np.sqrt(s.AR * Sw_new))
    return b_new, CL_target


def _exit_criteria(config) -> str:
    bits = [f"|ΔCD0| < {config.CD0_TOL:.0e}"]
    if config.MASS_CLOSURE:
        bits.append(f"|Δm_drone| < {config.MASS_TOL:.0e} kg")
    if config.SW_CLOSURE:
        bits.append(f"|ΔSw| < {config.SW_TOL:.0e} m²")
    return " AND ".join(bits)


def _last_deltas(config, dCD0: float, dM: float, dSw: float) -> str:
    bits = [f"|ΔCD0|={dCD0:.2e}"]
    if config.MASS_CLOSURE:
        bits.append(f"|Δm|={dM:.2e}")
    if config.SW_CLOSURE:
        bits.append(f"|ΔSw|={dSw:.2e}")
    return ", ".join(bits)


def _print_iter(
    it: int,
    *,
    p: _DesignPass,
    sizing: SizingResult,
    m_drone: float,
    x_cg: float,
    CL_target: float | None,
    dCD0: float,
    dM: float,
    dSw: float,
) -> None:
    cl_str = f"CL*={CL_target:.3f}  " if CL_target is not None else ""
    print(f"    iter {it:2d}: "
          f"CD0={p.drag.CD0:.6f}(Δ={dCD0:.1e})  "
          f"m_drone={m_drone:.3f}(Δ={dM:.1e})  "
          f"Sw={sizing.Sw:.4f}(Δ={dSw:.1e})  "
          f"b={sizing.inputs.b:.3f}  {cl_str}"
          f"x_cg={x_cg:.4f}  "
          f"t_w={p.struct.t_w * 1000:.2f}mm({p.struct.fail_mode_w[:4]})  "
          f"t_t={p.struct.t_t * 1000:.2f}mm({p.struct.fail_mode_t[:4]})  "
          f"m_batt={p.propulsion.battery_mass:.3f}  "
          f"fus={p.fus.length:.3f}×{p.fus.width:.3f}×{p.fus.height:.3f}")


def run_pipeline(config) -> PipelineResult:
    """Run the full design pipeline and return a bundled result.

    `config` is any module exposing SIZING, PROPULSION, FUSELAGE,
    CONTROL_SURFACE, STRUCTURE, AIRFOIL, TAIL_AIRFOIL, ALPHA_SWEEP_DEG,
    ALPHA_SWEEP_LOOP_DEG, MASS_CLOSURE, SW_CLOSURE, N_ITER_MAX,
    CD0_TOL, MASS_TOL, SW_TOL.
    """
    # ----- Step 0: resolve airfoil → read t/c from its geometry -----
    airfoil = resolve_airfoil(config.AIRFOIL)
    tc = airfoil_thickness_to_chord(airfoil)
    print(f"Airfoil: {airfoil}  (t/c = {tc:.4f})")
    sizing_inputs = config.SIZING

    # ----- Step 1: initial sizing with guessed Cd0 -----
    sizing = wing.run(sizing_inputs, t_over_c_root=tc)

    # ----- Step 1b: load airfoil polar once (Re from initial sizing) -----
    # Section Cd is weakly Re-sensitive over the iteration-induced Re drift, so
    # we don't re-run XFOIL each pass.
    Re_ref = sizing.rho * sizing.inputs.V_cruise * sizing.c / 1.7894e-5
    polar = get_airfoil_polar(
        airfoil, Re=Re_ref, M=0.0, alpha_range=(-5.0, 15.0, 0.5), use_cache=True
    )
    Cl_max = float(np.max(polar.Cl))
    print(f"Polar: Cl_alpha = {polar.Cl_alpha:.3f}/rad, "
          f"alpha_L0 = {np.degrees(polar.alpha_L0):.2f}°, Cl_max = {Cl_max:.3f}")

    # ----- Steps 2-4: iterate propulsion → fuselage → drag → mass → wing-area → sizing -----
    mode = ", ".join([
        "mass-closure" if config.MASS_CLOSURE else "fixed m_drone",
        "Sw-closure" if config.SW_CLOSURE else "fixed Sw",
    ])
    print(f">>> Iterating propulsion → fuselage → drag → mass → wing-area → sizing  [{mode}]")
    print(f"    exit: {_exit_criteria(config)}  (max {config.N_ITER_MAX} passes)")

    cd0_guess = sizing_inputs.Cd0
    m_drone_guess = sizing_inputs.m_drone_empty
    sw_guess = sizing.Sw
    print(f"    iter  0: Cd0={cd0_guess:.6f}  m_drone={m_drone_guess:.3f}  "
          f"Sw={sw_guess:.4f}  b={sizing_inputs.b:.3f}")

    cd0_history: list[float] = []
    mass_history: list[float] = []
    sw_history: list[float] = []
    cg_history: list[float] = []

    cd0_prev = cd0_guess
    m_prev = m_drone_guess
    sw_prev = sw_guess
    dCD0 = dM = dSw = float("inf")
    m_drone = m_drone_guess
    converged = False
    it = 0
    for it in range(1, config.N_ITER_MAX + 1):
        p = _run_design_pass(sizing, config=config, polar=polar, airfoil=airfoil)
        m_drone = p.masses["total"]
        x_cg = p.cg["overall"]

        # --- Wing-area closure: resize Sw to put op-point at max-L/D for drone+payload ---
        if config.SW_CLOSURE:
            m_eff = m_drone if config.MASS_CLOSURE else sizing_inputs.m_drone_empty
            b_new, CL_target = _sw_closure_b(
                sizing, p.drag, polar,
                m_drone_eff=m_eff,
                alpha_range=config.ALPHA_SWEEP_LOOP_DEG,
            )
        else:
            CL_target = None
            b_new = sizing_inputs.b

        # --- Feed CD0, b, (m_drone) back into sizing inputs and re-run wing sizing ---
        replace_kwargs: dict = {"Cd0": p.drag.CD0, "b": b_new}
        if config.MASS_CLOSURE:
            replace_kwargs["m_drone_empty"] = m_drone
        sizing_inputs = dataclasses.replace(sizing_inputs, **replace_kwargs)
        sizing = wing.run(sizing_inputs, t_over_c_root=tc)

        cd0_history.append(p.drag.CD0)
        mass_history.append(m_drone)
        sw_history.append(sizing.Sw)
        cg_history.append(x_cg)
        dCD0 = abs(p.drag.CD0 - cd0_prev)
        dM = abs(m_drone - m_prev)
        dSw = abs(sizing.Sw - sw_prev)

        _print_iter(it, p=p, sizing=sizing, m_drone=m_drone, x_cg=x_cg,
                    CL_target=CL_target, dCD0=dCD0, dM=dM, dSw=dSw)

        mass_ok = (dM < config.MASS_TOL) if config.MASS_CLOSURE else True
        sw_ok = (dSw < config.SW_TOL) if config.SW_CLOSURE else True
        if dCD0 < config.CD0_TOL and mass_ok and sw_ok:
            converged = True
            break
        cd0_prev = p.drag.CD0
        m_prev = m_drone
        sw_prev = sizing.Sw

    if converged:
        print(f"    ✓ Converged in {it} iterations ({_exit_criteria(config)}).")
        if not config.MASS_CLOSURE:
            print(f"      Buildup m_drone = {m_drone:.3f} kg vs requirement "
                  f"{sizing_inputs.m_drone_empty:.3f} kg.")
    else:
        print(f"    ✗ Did NOT converge after {config.N_ITER_MAX} iterations "
              f"(last {_last_deltas(config, dCD0, dM, dSw)}).")

    # Final pass so every sub-solver result matches the converged sizing.
    p = _run_design_pass(sizing, config=config, polar=polar, airfoil=airfoil)
    if np.isclose(p.fus.height, config.FUSELAGE.casing_factor * p.fus.battery_height):
        print("    WARNING: fuselage height is just casing_factor × battery_height; "
              "airfoil height is not being used for fuselage sizing.")

    # ----- Step 5: LLT at the required CL -----
    CL_req = sizing.CL
    llt = llt_at_cl(sizing, polar, CL_target=CL_req)
    max_Cl_local = float(np.max(np.abs(llt.Cl_local)))
    lift_achievable = max_Cl_local <= Cl_max

    # ----- Step 6: CL/CD sweep → drone-only and drone+payload polars -----
    CL_sweep, CD_wing_sweep = wing_drag_polar(sizing, polar, config.ALPHA_SWEEP_DEG)
    CD_nonwing = p.drag.CD0_tail + p.drag.CD0_fus
    CD_payload = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
        sizing.inputs.n_drones * sizing.Sw
    )
    CD_drone_sweep = CD_wing_sweep + CD_nonwing
    CD_full_sweep = CD_drone_sweep + CD_payload

    # ----- Step 7: tail trim loading + induced drag, then full drag estimate -----
    tail_loading = tail_drag_at_cruise(
        sizing, polar, llt,
        x_cg=p.cg["overall"],
        tail_airfoil=config.TAIL_AIRFOIL,
    )
    cd_i_tail = tail_loading["CD_i_tail_wing_ref"]
    CD_full_buildup = full_drag_estimate(sizing, p.drag, llt, cd_i_tail=cd_i_tail)

    # ----- Step 8: stability scissor line (once, post-convergence) -----
    y_cg = weights_mass.compute_y_cg(
        sizing=sizing, fus=p.fus, structure=p.struct,
        masses=p.masses, airfoil_path=airfoil,
    )
    scissor = compute_scissor_data(
        sizing, p.fus,
        wing_llt=llt,
        tail_llt=tail_loading["llt_tail"],
        x_cg_current=p.cg["overall"],
        y_cg=y_cg["overall"],
        Vh_V=p.drag.inputs.Vh_V,
    )

    return PipelineResult(
        airfoil=airfoil,
        tail_airfoil=config.TAIL_AIRFOIL,
        tc=tc,
        polar=polar,
        cl_max=Cl_max,
        sizing=sizing,
        propulsion=p.propulsion,
        fus=p.fus,
        drag=p.drag,
        struct=p.struct,
        control_surface=p.control_surface,
        masses=p.masses,
        cg=p.cg,
        cd0_history=cd0_history,
        mass_history=mass_history,
        sw_history=sw_history,
        cg_history=cg_history,
        cd0_guess=cd0_guess,
        m_drone_guess=m_drone_guess,
        sw_guess=sw_guess,
        llt=llt,
        cl_req=CL_req,
        max_cl_local=max_Cl_local,
        lift_achievable=lift_achievable,
        cl_sweep=CL_sweep,
        cd_drone_sweep=CD_drone_sweep,
        cd_full_sweep=CD_full_sweep,
        cd_full_buildup=CD_full_buildup,
        cd_payload=CD_payload,
        cd_i_tail=cd_i_tail,
        tail_loading=tail_loading,
        scissor=scissor,
    )
