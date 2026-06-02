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
from aerodynamics.stability import calculate_Sh_S
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
from sizing import aileron, elevator, fuselage, rudder, wing
from sizing.aileron import AileronResult
from sizing.elevator import ElevatorResult
from sizing.fuselage import FuselageResult
from sizing.rudder import RudderResult
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
    elevator: ElevatorResult
    rudder: RudderResult
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
    tail_polar: AirfoilPolar,
    airfoil: str,
) -> _DesignPass:
    propulsion = prop_sizing.run(sizing, config.PROPULSION)
    fus = fuselage.run(
        sizing,
        config.FUSELAGE,
        battery_volume=propulsion.battery_volume,
        airfoil_path=airfoil,
        battery_x=config.BATTERY_X,
    )
    # Aileron must be computed before rods and drag — both need the hinge x/c.
    control_surface = aileron.run(sizing, config.CONTROL_SURFACE, polar=polar)
    drag = estimate_cd0(
        sizing, fus, control_surface, propulsion,
        wing_airfoil=airfoil,
        tail_airfoil=config.TAIL_AIRFOIL,
    )
    struct = rods.run(
        sizing,
        control_surface,
        propulsion,
        airfoil,
        config.TAIL_AIRFOIL,
        config.STRUCTURE,
        tail_polar=tail_polar,
    )
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
        battery_x=config.BATTERY_X,
        materials=config.MATERIALS,
    )
    return _DesignPass(propulsion, fus, drag, struct, control_surface, masses, cg)


def _sw_closure_ar(
    sizing: SizingResult,
    drag: DragResult,
    polar: AirfoilPolar,
    *,
    m_drone_eff: float,
    alpha_range: tuple[float, float, int],
    cl_max_wing: float,
    V_stall: float,
) -> tuple[float, float, float, bool]:
    """Pick the aspect ratio that places the operating CL at max drone+payload L/D,
    clamped by the stall-speed bound.

    CL_stall_bound = (V_stall / V_cruise)^2 · CL_max_wing is the largest cruise
    CL that keeps the stall speed at-or-below V_stall. If the unconstrained
    max-L/D CL exceeds the bound, we operate at the bound and accept a
    sub-optimal L/D. `cl_max_wing` is the 3D wing CL_max
    (section Cl_max · AR/(AR+2)).

    Returns (AR_new, CL_target, CL_stall_bound, stall_binding).
    """
    s = sizing.inputs
    CL_sw, CD_wing_sw = wing_drag_polar(sizing, polar, alpha_range)
    CD_nonwing = drag.CD0_tail_h + drag.CD0_tail_h + drag.CD0_fus
    CD_payload = s.Cd_payload * s.S_payload / (s.n_drones * sizing.Sw)
    LD = CL_sw / (CD_wing_sw + CD_nonwing + CD_payload)
    CL_optLD = float(CL_sw[int(np.argmax(LD))])

    CL_stall_bound = (V_stall / s.V_cruise) ** 2 * cl_max_wing
    stall_binding = CL_optLD > CL_stall_bound
    CL_target = CL_stall_bound if stall_binding else CL_optLD

    W = (m_drone_eff + s.m_payload / s.n_drones) * 9.80665
    Sw_new = W / (sizing.q_cruise * CL_target)
    AR_new = float(sizing.inputs.b ** 2 / Sw_new)
    return AR_new, CL_target, CL_stall_bound, stall_binding


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
    stall_binding: bool,
    dCD0: float,
    dM: float,
    dSw: float,
) -> None:
    if CL_target is None:
        cl_str = ""
    else:
        tag = "stall" if stall_binding else "L/D"
        cl_str = f"CL*={CL_target:.3f}[{tag}]  "
    print(f"    iter {it:2d}: "
          f"CD0={p.drag.CD0:.6f}(Δ={dCD0:.1e})  "
          f"m_drone={m_drone:.3f}(Δ={dM:.1e})  "
          f"Sw={sizing.Sw:.4f}(Δ={dSw:.1e})  "
          f"b={sizing.inputs.b:.3f} AR={sizing.inputs.AR:.3f}  {cl_str}"
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
    # lh (tail moment arm, wing AC → tail AC) is taken straight from config;
    # the physical boom length L_boom is derived from lh, c_aileron_to_c_wing,
    # and the tail chord. Sh is the closure variable, seeded from the Vh
    # tail-volume estimate on this first pass and overwritten by the scissor
    # result on each subsequent loop iteration.
    sizing = wing.run(
        sizing_inputs,
        t_over_c_root=tc,
        c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
    )

    # ----- Step 1b: load airfoil polar once (Re from initial sizing) -----
    # Section Cd is weakly Re-sensitive over the iteration-induced Re drift, so
    # we don't re-run XFOIL each pass.
    Re_ref = sizing.rho * sizing.inputs.V_cruise * sizing.c / 1.7894e-5
    polar = get_airfoil_polar(
        airfoil, Re=Re_ref, M=0.0, alpha_range=(-5.0, 15.0, 0.5), use_cache=True
    )
    tail_polar = get_airfoil_polar(
        config.TAIL_AIRFOIL, Re=Re_ref, M=0.0, alpha_range=(-10.0, 10.0, 0.5), use_cache=True
    )
    Cl_max = float(np.max(polar.Cl))
    print(f"Polar: Cl_alpha = {polar.Cl_alpha:.3f}/rad, "
          f"alpha_L0 = {np.degrees(polar.alpha_L0):.2f}°, Cl_max = {Cl_max:.3f}")
    print(f"Tail polar: {tail_polar.name}, Cl_max = {float(np.max(tail_polar.Cl)):.3f}")

    # 3D wing CL_max: derate the 2D section value with the AR/(AR+2) Prandtl
    # correction (same Cl↔CL relation used in sizing/wing.py).
    AR = sizing_inputs.AR
    CL_max_wing = Cl_max * AR / (AR + 2.0)
    CL_stall_bound_init = (config.V_STALL / sizing_inputs.V_cruise) ** 2 * CL_max_wing
    print(f"Stall: V_stall = {config.V_STALL:.2f} m/s, "
          f"CL_max_wing = Cl_max · AR/(AR+2) = {CL_max_wing:.3f}  →  "
          f"CL_cruise_bound = (V_stall/V_cruise)² · CL_max_wing = {CL_stall_bound_init:.3f}")

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
        p = _run_design_pass(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
        )
        m_drone = p.masses["total"]
        x_cg = p.cg["overall"]

        # --- LLT at cruise CL + tail trim + stability scissor for this iteration's state ---
        llt = llt_at_cl(sizing, polar, CL_target=sizing.CL)
        tail_loading = tail_drag_at_cruise(
            sizing, polar, llt,
            x_cg=x_cg,
            tail_airfoil=config.TAIL_AIRFOIL,
        )
        y_cg = weights_mass.compute_y_cg(
            sizing=sizing, fus=p.fus, structure=p.struct,
            masses=p.masses, airfoil_path=airfoil, cg=p.cg,
        )
        scissor = compute_scissor_data(
            sizing, p.fus,
            wing_llt=llt,
            tail_llt=tail_loading["llt_tail"],
            x_cg_current=x_cg,
            y_cg=y_cg["overall"],
            Vh_V=p.struct.Vh_V,
        )

        Sh_S_new = calculate_Sh_S(x_cg=x_cg, x_ac=scissor.x_ac, c=scissor.c, l_h=scissor.l_h,
                          CL_h=scissor.CL_h, CL_A_h=scissor.CL_A_h, Cm_ac=scissor.Cm_ac,
                          Vh_V=scissor.Vh_V, CL_alpha_h=scissor.CL_alpha_h,
                          CL_alpha_A_h=scissor.CL_alpha_A_h, dep_da=scissor.dep_da, SM=scissor.SM)
        Sh_new = Sh_S_new * sizing.Sw

        # --- Wing-area closure: resize Sw to put op-point at max-L/D for drone+payload,
        #     clamped by the stall-speed bound ---
        if config.SW_CLOSURE:
            m_eff = m_drone if config.MASS_CLOSURE else sizing_inputs.m_drone_empty
            AR_new, CL_target, CL_stall_bound, stall_binding = _sw_closure_ar(
                sizing, p.drag, polar,
                m_drone_eff=m_eff,
                alpha_range=config.ALPHA_SWEEP_LOOP_DEG,
                cl_max_wing=CL_max_wing,
                V_stall=config.V_STALL,
            )
        else:
            CL_target = None
            CL_stall_bound = (config.V_STALL / sizing_inputs.V_cruise) ** 2 * CL_max_wing
            stall_binding = False
            AR_new = sizing_inputs.AR

        # --- Feed CD0, b, Sh, (m_drone) back into sizing inputs and re-run wing sizing ---
        # L_tail stays at its config value; Sh is driven by the scissor.
        replace_kwargs: dict = {"Cd0": p.drag.CD0, "AR": AR_new, "Sh": Sh_new}

        if config.MASS_CLOSURE:
            replace_kwargs["m_drone_empty"] = m_drone
        sizing_inputs = dataclasses.replace(sizing_inputs, **replace_kwargs)
        sizing = wing.run(
            sizing_inputs,
            t_over_c_root=tc,
            c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
        )

        cd0_history.append(p.drag.CD0)
        mass_history.append(m_drone)
        sw_history.append(sizing.Sw)
        cg_history.append(x_cg)
        dCD0 = abs(p.drag.CD0 - cd0_prev)
        dM = abs(m_drone - m_prev)
        dSw = abs(sizing.Sw - sw_prev)

        _print_iter(it, p=p, sizing=sizing, m_drone=m_drone, x_cg=x_cg,
                    CL_target=CL_target, stall_binding=stall_binding,
                    dCD0=dCD0, dM=dM, dSw=dSw)

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
    p = _run_design_pass(
        sizing,
        config=config,
        polar=polar,
        tail_polar=tail_polar,
        airfoil=airfoil,
    )
    if np.isclose(p.fus.height, config.FUSELAGE.casing_factor * p.fus.battery_height):
        print("    WARNING: fuselage height is just casing_factor × battery_height; "
              "airfoil height is not being used for fuselage sizing.")

    # Stall check on the converged design. When the Sw closure is active and the
    # stall bound is binding, V_stall_actual equals V_stall by construction;
    # a 1e-3 m/s tolerance absorbs floating-point noise.
    W_loaded = sizing.m_drone_loaded * 9.80665
    V_stall_actual = float(np.sqrt(2 * W_loaded / (sizing.rho * sizing.Sw * CL_max_wing)))
    margin = config.V_STALL - V_stall_actual
    flag = "OK" if margin >= -1.0e-3 else "FAIL"
    print(f"    Stall check: V_stall_actual = {V_stall_actual:.3f} m/s  "
          f"(requirement ≤ {config.V_STALL:.3f} m/s, margin = {margin:+.3f} m/s)  [{flag}]")

    # ----- Step 5: LLT at the required CL -----
    CL_req = sizing.CL
    llt = llt_at_cl(sizing, polar, CL_target=CL_req)
    max_Cl_local = float(np.max(np.abs(llt.Cl_local)))
    lift_achievable = max_Cl_local <= Cl_max

    # ----- Step 6: CL/CD sweep → drone-only and drone+payload polars -----
    CL_sweep, CD_wing_sweep = wing_drag_polar(sizing, polar, config.ALPHA_SWEEP_DEG)
    CD_nonwing = p.drag.CD0_tail_h + p.drag.CD0_tail_v + p.drag.CD0_fus
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

    # ----- Step 8: stability scissor line on the final consistent state -----
    y_cg = weights_mass.compute_y_cg(
        sizing=sizing, fus=p.fus, structure=p.struct,
        masses=p.masses, airfoil_path=airfoil, cg=p.cg,
    )
    scissor = compute_scissor_data(
        sizing, p.fus,
        wing_llt=llt,
        tail_llt=tail_loading["llt_tail"],
        x_cg_current=p.cg["overall"],
        y_cg=y_cg["overall"],
        Vh_V=p.struct.inputs.Vh_V,
    )

    # ----- Step 9: elevator + rudder control-surface sizing on the final state -----
    elevator_result = elevator.run(
        sizing,
        scissor,
        llt,
        p.propulsion,
        polar,
        tail_polar,
        y_cg,
        config.ELEVATOR,
    )

    rudder_result = rudder.run(
        sizing, scissor, p.fus, config.V_STALL, config.RUDDER,
        x_cg=p.cg["overall"],
    )

    F_tail = (
        sizing.Sh
        * abs(-0.35 * sizing.inputs.ARt ** (1.0 / 3.0))
        * sizing.q_cruise
        * config.STRUCTURE.Vh_V
        * config.STRUCTURE.safety_factor
    )
    struct = rods.apply_torsion_check(
        rod=p.struct,
        rudder_hinge_moment=rudder_result.hinge_moment.H,
        bending_force=F_tail,
        boom_length=sizing.L_boom,
        safety_factor=config.STRUCTURE.safety_factor,
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
        struct=struct,
        control_surface=p.control_surface,
        elevator=elevator_result,
        rudder=rudder_result,
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
