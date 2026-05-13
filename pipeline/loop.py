"""Sizing↔propulsion↔fuselage↔drag↔mass↔wing-area convergence loop.

`run_pipeline(config)` reads design knobs from a config module (any module
exposing the module-level constants in `pipeline.config`) and returns a
`PipelineResult` bundling every value the report / plot consumers need.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from aerodynamics.airfoil_geometry import airfoil_thickness_to_chord
from aerodynamics.airfoil_polar import AirfoilPolar, get_airfoil_polar
from aerodynamics.drag_buildup import DragResult
from aerodynamics.llt import LLTResult
from pipeline.helpers import (
    estimate_cd0,
    full_drag_estimate,
    llt_at_cl,
    resolve_airfoil,
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
    # Reused inside the loop for control-surface section Cd, and after the loop for LLT.
    # Section Cd is weakly Re-sensitive over the iteration-induced Re drift, so we
    # don't re-run XFOIL each pass.
    Re_ref = sizing.rho * sizing.inputs.V_cruise * sizing.c / 1.7894e-5
    polar = get_airfoil_polar(
        airfoil, Re=Re_ref, M=0.0, alpha_range=(-5.0, 15.0, 0.5), use_cache=True
    )
    Cl_max = float(np.max(polar.Cl))
    print(f"Polar: Cl_alpha = {polar.Cl_alpha:.3f}/rad, "
          f"alpha_L0 = {np.degrees(polar.alpha_L0):.2f}°, Cl_max = {Cl_max:.3f}")

    # ----- Steps 2-4: iterate propulsion → fuselage → drag → mass → wing-area → sizing -----
    mode_bits = []
    mode_bits.append("mass-closure" if config.MASS_CLOSURE else "fixed m_drone")
    mode_bits.append("Sw-closure" if config.SW_CLOSURE else "fixed Sw")
    print(f">>> Iterating propulsion → fuselage → drag → mass → wing-area → sizing  "
          f"[{', '.join(mode_bits)}]")
    exit_bits = [f"|ΔCD0| < {config.CD0_TOL:.0e}"]
    if config.MASS_CLOSURE:
        exit_bits.append(f"|Δm_drone| < {config.MASS_TOL:.0e} kg")
    if config.SW_CLOSURE:
        exit_bits.append(f"|ΔSw| < {config.SW_TOL:.0e} m²")
    print(f"    exit: {' AND '.join(exit_bits)}  (max {config.N_ITER_MAX} passes)")
    Cd0_guess = sizing_inputs.Cd0
    m_drone_guess = sizing_inputs.m_drone_empty
    Sw_guess = sizing.Sw
    print(f"    iter  0: Cd0={Cd0_guess:.6f}  m_drone={m_drone_guess:.3f}  "
          f"Sw={Sw_guess:.4f}  b={sizing_inputs.b:.3f}")

    CD0_history: list[float] = []
    mass_history: list[float] = []
    Sw_history: list[float] = []
    cg_history: list[float] = []
    CD0_prev = Cd0_guess
    m_prev = m_drone_guess
    Sw_prev = Sw_guess
    converged = False
    # Initialise loop locals so they survive a zero-iteration max (defensive).
    dCD0 = dM = dSw = float("inf")
    m_drone = m_drone_guess
    propulsion = fus = drag = struct = control_surface = None  # type: ignore[assignment]
    masses = cg = {}
    it = -1
    for it in range(config.N_ITER_MAX):
        propulsion = prop_sizing.run(sizing, config.PROPULSION)
        fus = fuselage.run(
            sizing,
            config.FUSELAGE,
            battery_volume=propulsion.battery_volume,
            airfoil_path=airfoil,
        )
        if np.isclose(fus.height, config.FUSELAGE.casing_factor * fus.battery_height):
            print(
                "    WARNING: fuselage height is just casing_factor × battery_height; "
                "airfoil height is not being used for fuselage sizing."
            )
        drag = estimate_cd0(sizing, fus, wing_airfoil=airfoil, tail_airfoil=config.TAIL_AIRFOIL)
        struct = rods.run(sizing, config.STRUCTURE)
        masses = weights_mass.total_mass(
            sizing=sizing,
            propulsion=propulsion,
            structure=struct,
            fus=fus,
            airfoil_path=airfoil,
            tail_airfoil_path=config.TAIL_AIRFOIL,
        )
        m_drone = masses["total"]
        cg = weights_mass.compute_cg(
            sizing=sizing,
            propulsion=propulsion,
            structure=struct,
            fus=fus,
            airfoil_path=airfoil,
            tail_airfoil_path=config.TAIL_AIRFOIL,
        )
        x_cg = cg["overall"]

        # --- Wing-area closure: resize Sw to put op-point at max-L/D for drone+payload ---
        if config.SW_CLOSURE:
            CL_sw, CD_wing_sw = wing_drag_polar(sizing, polar, config.ALPHA_SWEEP_LOOP_DEG)
            CD_nonwing = drag.CD0_tail + drag.CD0_fus
            CD_payload_sw = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
                sizing.inputs.n_drones * sizing.Sw
            )
            LD_full_sw = CL_sw / (CD_wing_sw + CD_nonwing + CD_payload_sw)
            CL_target = float(CL_sw[int(np.argmax(LD_full_sw))])
            m_eff = m_drone if config.MASS_CLOSURE else sizing_inputs.m_drone_empty
            W = (m_eff + sizing_inputs.m_payload / sizing_inputs.n_drones) * 9.80665
            Sw_new = W / (sizing.q_cruise * CL_target)
            b_new = float(np.sqrt(sizing_inputs.AR * Sw_new))
        else:
            CL_target = None
            b_new = sizing_inputs.b

        # --- Build updated sizing inputs ---
        replace_kwargs: dict = {"Cd0": drag.CD0, "b": b_new}
        if config.MASS_CLOSURE:
            replace_kwargs["m_drone_empty"] = m_drone
        sizing_inputs = dataclasses.replace(sizing_inputs, **replace_kwargs)
        sizing = wing.run(sizing_inputs, t_over_c_root=tc)
        control_surface = aileron.run(sizing, config.CONTROL_SURFACE, polar=polar)

        CD0_history.append(drag.CD0)
        mass_history.append(m_drone)
        Sw_history.append(sizing.Sw)
        cg_history.append(x_cg)
        dCD0 = abs(drag.CD0 - CD0_prev)
        dM = abs(m_drone - m_prev)
        dSw = abs(sizing.Sw - Sw_prev)
        cl_str = f"CL*={CL_target:.3f}  " if CL_target is not None else ""
        print(f"    iter {it + 1:2d}: "
              f"CD0={drag.CD0:.6f}(Δ={dCD0:.1e})  "
              f"m_drone={m_drone:.3f}(Δ={dM:.1e})  "
              f"Sw={sizing.Sw:.4f}(Δ={dSw:.1e})  "
              f"b={sizing.inputs.b:.3f}  {cl_str}"
              f"x_cg={x_cg:.4f}  "
              f"t_w={struct.t_w * 1000:.2f}mm({struct.fail_mode_w[:4]})  "
              f"t_t={struct.t_t * 1000:.2f}mm({struct.fail_mode_t[:4]})  "
              f"m_batt={propulsion.battery_mass:.3f}  "
              f"fus={fus.length:.3f}×{fus.width:.3f}×{fus.height:.3f}")
        mass_ok = (dM < config.MASS_TOL) if config.MASS_CLOSURE else True
        sw_ok = (dSw < config.SW_TOL) if config.SW_CLOSURE else True
        if dCD0 < config.CD0_TOL and mass_ok and sw_ok:
            converged = True
            break
        CD0_prev = drag.CD0
        m_prev = m_drone
        Sw_prev = sizing.Sw

    if converged:
        ok_bits = [f"|ΔCD0| < {config.CD0_TOL:.0e}"]
        if config.MASS_CLOSURE:
            ok_bits.append(f"|Δm| < {config.MASS_TOL:.0e} kg")
        if config.SW_CLOSURE:
            ok_bits.append(f"|ΔSw| < {config.SW_TOL:.0e} m²")
        print(f"    ✓ Converged in {it + 1} iterations ({', '.join(ok_bits)}).")
        if not config.MASS_CLOSURE:
            print(f"      Buildup m_drone = {m_drone:.3f} kg vs requirement "
                  f"{sizing_inputs.m_drone_empty:.3f} kg.")
    else:
        last_bits = [f"|ΔCD0|={dCD0:.2e}"]
        if config.MASS_CLOSURE:
            last_bits.append(f"|Δm|={dM:.2e}")
        if config.SW_CLOSURE:
            last_bits.append(f"|ΔSw|={dSw:.2e}")
        print(f"    ✗ Did NOT converge after {config.N_ITER_MAX} iterations "
              f"(last {', '.join(last_bits)}).")

    # Final pass with the converged Cd0 so propulsion/fus/drag match the latest sizing.
    propulsion = prop_sizing.run(sizing, config.PROPULSION)
    fus = fuselage.run(
        sizing,
        config.FUSELAGE,
        battery_volume=propulsion.battery_volume,
        airfoil_path=airfoil,
    )
    drag = estimate_cd0(sizing, fus, wing_airfoil=airfoil, tail_airfoil=config.TAIL_AIRFOIL)
    struct = rods.run(sizing, config.STRUCTURE)
    control_surface = aileron.run(sizing, config.CONTROL_SURFACE, polar=polar)
    masses = weights_mass.total_mass(
        sizing=sizing,
        propulsion=propulsion,
        structure=struct,
        fus=fus,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
    )
    cg = weights_mass.compute_cg(
        sizing=sizing,
        propulsion=propulsion,
        structure=struct,
        fus=fus,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
    )

    # ----- Step 5: LLT at the required CL -----
    CL_req = sizing.CL
    llt = llt_at_cl(sizing, polar, CL_target=CL_req)
    max_Cl_local = float(np.max(np.abs(llt.Cl_local)))
    lift_achievable = max_Cl_local <= Cl_max

    # ----- Step 6: CL/CD sweep → drone-only and drone+payload polars -----
    CL_sweep, CD_wing_sweep = wing_drag_polar(sizing, polar, config.ALPHA_SWEEP_DEG)
    CD_nonwing = drag.CD0_tail + drag.CD0_fus
    CD_payload = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
        sizing.inputs.n_drones * sizing.Sw
    )
    CD_drone_sweep = CD_wing_sweep + CD_nonwing
    CD_full_sweep = CD_drone_sweep + CD_payload

    # ----- Step 7: full drag estimate = CD0 buildup + induced + payload -----
    CD_full_buildup = full_drag_estimate(sizing, drag, llt)

    return PipelineResult(
        airfoil=airfoil,
        tail_airfoil=config.TAIL_AIRFOIL,
        tc=tc,
        polar=polar,
        cl_max=Cl_max,
        sizing=sizing,
        propulsion=propulsion,
        fus=fus,
        drag=drag,
        struct=struct,
        control_surface=control_surface,
        masses=masses,
        cg=cg,
        cd0_history=CD0_history,
        mass_history=mass_history,
        sw_history=Sw_history,
        cg_history=cg_history,
        cd0_guess=Cd0_guess,
        m_drone_guess=m_drone_guess,
        sw_guess=Sw_guess,
        llt=llt,
        cl_req=CL_req,
        max_cl_local=max_Cl_local,
        lift_achievable=lift_achievable,
        cl_sweep=CL_sweep,
        cd_drone_sweep=CD_drone_sweep,
        cd_full_sweep=CD_full_sweep,
        cd_full_buildup=CD_full_buildup,
        cd_payload=CD_payload,
    )
