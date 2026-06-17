"""Sizing↔propulsion↔fuselage↔drag↔mass↔wing-area convergence loop.

`run_pipeline(config)` reads design knobs from a config module (any module
exposing the module-level constants in `pipeline.config`) and returns a
`PipelineResult` bundling every value the report / plot consumers need.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from aerodynamics.airfoil_geometry import AirfoilGeometry, airfoil_thickness_to_chord
from aerodynamics.airfoil_polar import AirfoilPolar, get_airfoil_polar
from aerodynamics.drag_buildup import DragResult
from aerodynamics.llt import FlightCondition, LLTResult, WingGeometry, solve_llt
from pipeline.helpers import (
    ScissorData,
    compute_cm_thrust,
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
    cl_max_wing: float

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
    failure_lift_achievable: bool
    cl_sweep: np.ndarray
    cd_drone_sweep: np.ndarray
    cd_full_sweep: np.ndarray
    cd_full_buildup: float
    cd_payload: float
    cd_i_tail: float       # tail induced drag, wing-area reference
    tail_loading: dict     # CL_tail, CD_i_tail (tail ref), e_tail, AR_tail
    scissor: ScissorData   # stability-line scissor plot data
    rib_checks: tuple[dict[str, float | str | bool], ...]
    battery_y0: float      # [m] battery bottom used for final vertical trim
    y_cg: dict[str, float] # final vertical CG breakdown


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


@dataclass
class _ScissorState:
    """Scissor-related outputs for a design pass."""
    llt: LLTResult
    tail_loading: dict
    y_cg: dict[str, float]
    scissor: ScissorData
    x_target: float
    ShS_target: float


def _boom_root_x_from_battery(battery_x: float | None, fus: FuselageResult | None) -> float | None:
    if battery_x is None or fus is None:
        return None
    return float(battery_x) - 0.5 * fus.battery_length


def _sizing_inputs_with_boom_root(
    inputs: wing.SizingInputs,
    battery_x: float | None,
    fus: FuselageResult | None,
    **replace_kwargs,
) -> wing.SizingInputs:
    boom_root_x = _boom_root_x_from_battery(battery_x, fus)
    if boom_root_x is not None:
        replace_kwargs["boom_root_x"] = boom_root_x
    return dataclasses.replace(inputs, **replace_kwargs)


def _run_design_pass(
    sizing: SizingResult,
    *,
    config,
    polar: AirfoilPolar,
    tail_polar: AirfoilPolar,
    airfoil: str,
    battery_x: float | None,
) -> _DesignPass:
    propulsion = prop_sizing.run(sizing, config.PROPULSION)
    q_control = 0.5 * sizing.rho * config.V_STALL ** 2
    V_structural = sizing.inputs.v_max
    q_structural = 0.5 * sizing.rho * V_structural ** 2
    lift_one_drone_failure_structural = (
        sizing.CL_one_drone_failure * q_structural * sizing.Sw
    )
    lift_structural_increment = (
        lift_one_drone_failure_structural - sizing.lift_one_drone_failure
    )

    # Aileron must be computed before rods — rods need the hinge x/c.
    control_surface = aileron.run(sizing, config.CONTROL_SURFACE, polar=polar)

    # Rods before fuselage: fuselage needs tube OD and tube aft-x to set its
    # height and length.  Rods only need sizing + control_surface + propulsion,
    # so there is no circular dependency here.
    struct = rods.run(
        sizing,
        control_surface,
        propulsion,
        airfoil,
        config.TAIL_AIRFOIL,
        config.STRUCTURE,
        tail_polar=tail_polar,
    )

    # Boom/fuselage geometry. The old green structural sleeve is gone; the
    # tail boom sits near the fuselage floor, with the wing rods directly above.
    tube_outer_diameter = struct.d_t
    x_rod_aileron = (1.0 - control_surface.inputs.c_aileron_to_c_wing) * sizing.c_root
    wing_airfoil_geom = AirfoilGeometry(airfoil)
    _, x_max_tc = wing_airfoil_geom.compute_maximum_thickness()
    x_rod_spar = x_max_tc * sizing.c_root
    wing_section_height = (
        float(np.max(wing_airfoil_geom.polygon[:, 1]) - np.min(wing_airfoil_geom.polygon[:, 1]))
        * sizing.c_root
    )
    airfoil_y_offset = -float(np.min(wing_airfoil_geom.polygon[:, 1])) * sizing.c_root
    _, y_up_s, y_lo_s = wing_airfoil_geom.compute_thickness(x_rod_spar / sizing.c_root)
    _, y_up_a, y_lo_a = wing_airfoil_geom.compute_thickness(x_rod_aileron / sizing.c_root)
    y_rod_spar = 0.5 * (y_up_s + y_lo_s) * sizing.c_root + airfoil_y_offset
    y_rod_aileron = 0.5 * (y_up_a + y_lo_a) * sizing.c_root + airfoil_y_offset
    lowest_wing_rod_bottom = min(
        y_rod_spar - struct.d_spar / 2.0,
        y_rod_aileron - struct.d_aileron / 2.0,
    )
    tail_boom_bottom = config.FUSELAGE.min_foam_floor
    tail_boom_top = tail_boom_bottom + struct.d_t
    wing_y_shift = tail_boom_top - lowest_wing_rod_bottom
    rod_upper = max(
        y_rod_spar + wing_y_shift + struct.d_spar / 2.0,
        y_rod_aileron + wing_y_shift + struct.d_aileron / 2.0,
    )
    wing_top_y = wing_y_shift + wing_section_height
    tube_top_y = max(tail_boom_top, rod_upper, wing_top_y)
    battery_bottom_y = tail_boom_top
    tube_front_x = x_rod_spar - config.FUSELAGE.tube_tail_overlap
    tube_back_x = x_rod_aileron + config.FUSELAGE.tube_tail_overlap
    tube_length = tube_back_x - tube_front_x

    fus = fuselage.run(
        sizing,
        config.FUSELAGE,
        battery_volume=propulsion.battery_volume,
        battery_x=battery_x,
        battery_mass=propulsion.battery_mass,
        battery_bottom_y=battery_bottom_y,
        tube_top_y=tube_top_y,
        tube_back_x=tube_back_x,
        tube_outer_diameter=tube_outer_diameter,
        tube_length=tube_length,
        spar_rod_diameter=struct.d_spar,
        aileron_rod_diameter=struct.d_aileron,
        tail_rod_diameter=struct.d_t,
        wing_section_height=wing_section_height,
        structural_lift_increment=lift_structural_increment,
        x_front_spar=x_rod_spar,
        n_active_drones=sizing.inputs.n_drones - 1,
        m_total_failure=(
            sizing.inputs.n_drones * sizing.inputs.m_drone_empty
            + sizing.inputs.m_payload
        ),
    )

    drag = estimate_cd0(
        sizing, fus, struct,
        wing_airfoil=airfoil,
        tail_airfoil=config.TAIL_AIRFOIL,
    )
    masses = weights_mass.total_mass(
        sizing=sizing,
        propulsion=propulsion,
        structure=struct,
        fus=fus,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
        aileron=control_surface,
        battery_x=battery_x,
        materials=config.MATERIALS,
        rib_inputs=config.RIBS,
        rudder_inputs=config.RUDDER,
    )
    cg = weights_mass.compute_cg(
        sizing=sizing,
        propulsion=propulsion,
        structure=struct,
        fus=fus,
        aileron=control_surface,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
        battery_x=battery_x,
        materials=config.MATERIALS,
        rib_inputs=config.RIBS,
        rudder_inputs=config.RUDDER,
    )
    y_cg_for_aileron = weights_mass.compute_y_cg(
        sizing=sizing,
        fus=fus,
        structure=struct,
        masses=masses,
        airfoil_path=airfoil,
        cg=cg,
        rib_inputs=config.RIBS,
    )
    control_surface = aileron.run(
        sizing,
        config.CONTROL_SURFACE,
        polar=polar,
        y_cg=y_cg_for_aileron,
        q_sizing=q_control,
        payload_max_tension=config.PAYLOAD_MAX_TENSION,
    )
    return _DesignPass(propulsion, fus, drag, struct, control_surface, masses, cg)


def _scissor_intersection(scissor: ScissorData) -> tuple[float, float]:
    """Return the lowest stability/controllability intersection on the scissor plot."""
    diff = scissor.ShS_stab - scissor.ShS_ctrl
    candidates: list[tuple[float, float]] = []

    for k in range(len(diff) - 1):
        d0 = diff[k]
        d1 = diff[k + 1]
        if d0 == 0.0:
            x = float(scissor.x_cg[k])
            y = float(scissor.ShS_stab[k])
            candidates.append((x, y))
        elif d0 * d1 < 0.0:
            frac = -d0 / (d1 - d0)
            x = float(scissor.x_cg[k] + frac * (scissor.x_cg[k + 1] - scissor.x_cg[k]))
            y = float(scissor.ShS_stab[k] + frac * (scissor.ShS_stab[k + 1] - scissor.ShS_stab[k]))
            candidates.append((x, y))

    if diff[-1] == 0.0:
        candidates.append((float(scissor.x_cg[-1]), float(scissor.ShS_stab[-1])))

    if candidates:
        return min(candidates, key=lambda xy: xy[1])

    k = int(np.argmin(np.abs(diff)))
    x = float(scissor.x_cg[k])
    y = float(max(scissor.ShS_stab[k], scissor.ShS_ctrl[k]))
    return x, y


def _stability_x_at_ShS(
    scissor: ScissorData,
    ShS_horizontal: float,
    *,
    reference_x: float | None = None,
) -> tuple[float, float]:
    """Return where the current horizontal Sh/S line crosses the stability curve."""
    diff = scissor.ShS_stab - ShS_horizontal
    candidates: list[float] = []

    for k in range(len(diff) - 1):
        d0 = float(diff[k])
        d1 = float(diff[k + 1])
        if d0 == 0.0:
            candidates.append(float(scissor.x_cg[k]))
        elif d0 * d1 < 0.0:
            frac = -d0 / (d1 - d0)
            x = float(scissor.x_cg[k] + frac * (scissor.x_cg[k + 1] - scissor.x_cg[k]))
            candidates.append(x)

    if float(diff[-1]) == 0.0:
        candidates.append(float(scissor.x_cg[-1]))

    if candidates:
        if reference_x is None:
            return candidates[0], float(ShS_horizontal)
        return min(candidates, key=lambda x: abs(x - reference_x)), float(ShS_horizontal)

    k = int(np.argmin(np.abs(diff)))
    return float(scissor.x_cg[k]), float(ShS_horizontal)


def _payload_trim_moment_for_scissor(config, scissor: ScissorData) -> float:
    """Payload moment used by the scissor controllability line."""
    if config.ELEVATOR.trim_mode.lower() != "payload_attachment":
        return 0.0
    return -(
        scissor.CL_A_h * (scissor.x_cg_current - scissor.x_ac) / scissor.c
        + scissor.Cm_ac
        + scissor.Cm_thrust
    )


def _scissor_state_for_pass(
    sizing: SizingResult,
    p: _DesignPass,
    *,
    config,
    polar: AirfoilPolar,
    airfoil: str,
) -> _ScissorState:
    x_cg = p.cg["overall"]
    llt = llt_at_cl(sizing, polar, CL_target=sizing.CL)
    payload_trim_mode = config.ELEVATOR.trim_mode.lower() == "payload_attachment"
    CL_tail_trim = 1.0e-3 if payload_trim_mode else None
    tail_loading = tail_drag_at_cruise(
        sizing, polar, llt,
        x_cg=x_cg,
        tail_airfoil=config.TAIL_AIRFOIL,
        CL_tail=CL_tail_trim,
    )
    tail_loading_scissor = tail_loading
    if payload_trim_mode:
        tail_loading_scissor = tail_drag_at_cruise(
            sizing, polar, llt,
            x_cg=x_cg,
            tail_airfoil=config.TAIL_AIRFOIL,
            CL_tail=0.5,
        )
    y_cg = weights_mass.compute_y_cg(
        sizing=sizing, fus=p.fus, structure=p.struct,
        masses=p.masses, airfoil_path=airfoil, cg=p.cg,
        rib_inputs=config.RIBS,
    )

    y_thrust_ref = y_cg["wing_ac"]
    Z_T_front = y_cg["motors"]     - y_thrust_ref
    Z_T_back  = y_cg["motor_back"] - y_thrust_ref
    Cm_front, Cm_back = compute_cm_thrust(
        p.propulsion.thrust_cruise_per_prop,
        Z_T_front, Z_T_back,
        sizing.q_cruise, sizing.Sw, sizing.c,
    )

    # eta_h is the dynamic pressure ratio; stability.py expects a velocity
    # ratio (Vh_V) and squares it internally, so pass sqrt(eta_h).
    Vh_V = np.sqrt(p.struct.eta_h)

    scissor = compute_scissor_data(
        sizing, p.fus,
        wing_llt=llt,
        tail_llt=tail_loading_scissor["llt_tail"],
        x_cg_current=x_cg,
        y_cg=y_cg["overall"],
        Cm_thrust=Cm_front + Cm_back,
        Vh_V=Vh_V,
    )
    Cm_payload = _payload_trim_moment_for_scissor(config, scissor)
    if Cm_payload != 0.0:
        scissor = compute_scissor_data(
            sizing, p.fus,
            wing_llt=llt,
            tail_llt=tail_loading_scissor["llt_tail"],
            x_cg_current=x_cg,
            y_cg=y_cg["overall"],
            Cm_thrust=Cm_front + Cm_back,
            Cm_payload=Cm_payload,
            Vh_V=Vh_V,
        )
    ShS_current = sizing.Sh / sizing.Sw
    x_target, _ = _stability_x_at_ShS(
        scissor,
        ShS_current,
        reference_x=x_cg,
    )
    if config.ELEVATOR.trim_mode.lower() == "payload_attachment":
        ShS_target = float(max(
            np.interp(x_cg, scissor.x_cg, scissor.ShS_stab),
            np.interp(x_cg, scissor.x_cg, scissor.ShS_ctrl),
        ))
    else:
        _, ShS_target = _scissor_intersection(scissor)
    return _ScissorState(
        llt=llt,
        tail_loading=tail_loading,
        y_cg=y_cg,
        scissor=scissor,
        x_target=x_target,
        ShS_target=ShS_target,
    )


def _evaluate_battery_x(
    sizing: SizingResult,
    *,
    config,
    polar: AirfoilPolar,
    tail_polar: AirfoilPolar,
    airfoil: str,
    battery_x: float | None,
) -> tuple[_DesignPass, _ScissorState, float]:
    p = _run_design_pass(
        sizing,
        config=config,
        polar=polar,
        tail_polar=tail_polar,
        airfoil=airfoil,
        battery_x=battery_x,
    )
    state = _scissor_state_for_pass(sizing, p, config=config, polar=polar, airfoil=airfoil)
    return p, state, p.cg["overall"] - state.x_target


def _optimize_battery_x_for_scissor(
    sizing: SizingResult,
    *,
    config,
    polar: AirfoilPolar,
    tail_polar: AirfoilPolar,
    airfoil: str,
    battery_x_initial: float | None,
    tol: float = 1.0e-4,
    max_iter: int = 8,
) -> tuple[float, _DesignPass, _ScissorState, float]:
    """Move the battery until CG lies on the stability curve at the current Sh/S."""
    battery_x = battery_x_initial
    p, state, error = _evaluate_battery_x(
        sizing,
        config=config,
        polar=polar,
        tail_polar=tail_polar,
        airfoil=airfoil,
        battery_x=battery_x,
    )
    if battery_x is None:
        battery_x = p.cg["battery"]

    best = (abs(error), float(battery_x), p, state, error)
    prev_x: float | None = None
    prev_error: float | None = None
    max_step = max(0.25 * sizing.c, 0.05)

    for _ in range(max_iter):
        if abs(error) < tol:
            break

        if prev_x is None or prev_error is None:
            probe_step = max(0.05 * sizing.c, 0.01)
            probe_x = float(battery_x) + probe_step
            p_probe, state_probe, probe_error = _evaluate_battery_x(
                sizing,
                config=config,
                polar=polar,
                tail_polar=tail_polar,
                airfoil=airfoil,
                battery_x=probe_x,
            )
            if abs(probe_error) < best[0]:
                best = (abs(probe_error), probe_x, p_probe, state_probe, probe_error)
            derivative = (probe_error - error) / probe_step
            prev_x, prev_error = float(battery_x), error
        else:
            derivative = (error - prev_error) / (float(battery_x) - prev_x)

        if not np.isfinite(derivative) or abs(derivative) < 1.0e-6:
            next_x = float(battery_x) - np.sign(error) * max_step
        else:
            step = -error / derivative
            step = float(np.clip(step, -max_step, max_step))
            next_x = float(battery_x) + step

        prev_x, prev_error = float(battery_x), error
        battery_x = next_x
        p, state, error = _evaluate_battery_x(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
            battery_x=battery_x,
        )
        if abs(error) < best[0]:
            best = (abs(error), float(battery_x), p, state, error)

    _, battery_x, p, state, error = best
    return battery_x, p, state, error


def _retune_tail_area_and_battery_for_scissor(
    sizing: SizingResult,
    *,
    config,
    polar: AirfoilPolar,
    tail_polar: AirfoilPolar,
    airfoil: str,
    t_over_c_root: float,
    battery_x_initial: float | None,
    tol_x: float = 1.0e-4,
    tol_ShS: float = 1.0e-4,
    max_iter: int = 8,
) -> tuple[SizingResult, float, _DesignPass, _ScissorState, float, float]:
    """Couple battery-x with the stability line and Sh/S with the scissor minimum."""
    battery_x = battery_x_initial
    p: _DesignPass | None = None
    state: _ScissorState | None = None
    cg_error = float("inf")
    ShS_error = float("inf")

    for _ in range(max_iter):
        battery_x, p, state, cg_error = _optimize_battery_x_for_scissor(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
            battery_x_initial=battery_x,
            tol=tol_x,
        )
        ShS_current = sizing.Sh / sizing.Sw
        ShS_error = ShS_current - state.ShS_target
        if abs(cg_error) <= tol_x and abs(ShS_error) <= tol_ShS:
            break

        sizing_inputs = _sizing_inputs_with_boom_root(
            sizing.inputs,
            battery_x,
            p.fus,
            Sh=state.ShS_target * sizing.Sw,
        )
        sizing = wing.run(
            sizing_inputs,
            t_over_c_root=t_over_c_root,
            c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
        )

    assert p is not None and state is not None
    return sizing, battery_x, p, state, cg_error, ShS_error


def _final_scissor_for_y_cg(
    sizing: SizingResult,
    p: _DesignPass,
    *,
    config,
    llt: LLTResult,
    tail_loading: dict,
    y_cg: dict[str, float],
    propulsion: PropulsionResult,
) -> ScissorData:
    y_thrust_ref = y_cg["wing_ac"]
    Z_T_front = y_cg["motors"]     - y_thrust_ref
    Z_T_back  = y_cg["motor_back"] - y_thrust_ref
    Cm_front, Cm_back = compute_cm_thrust(
        propulsion.thrust_cruise_per_prop,
        Z_T_front, Z_T_back,
        sizing.q_cruise, sizing.Sw, sizing.c,
    )

    # eta_h is the dynamic pressure ratio; stability.py expects a velocity
    # ratio (Vh_V) and squares it internally, so pass sqrt(eta_h).
    Vh_V = np.sqrt(p.struct.inputs.eta_h)

    scissor = compute_scissor_data(
        sizing, p.fus,
        wing_llt=llt,
        tail_llt=tail_loading["llt_tail"],
        x_cg_current=p.cg["overall"],
        y_cg=y_cg["overall"],
        Cm_thrust=Cm_front + Cm_back,
        Vh_V=Vh_V,
    )
    Cm_payload = _payload_trim_moment_for_scissor(config, scissor)
    if Cm_payload != 0.0:
        scissor = compute_scissor_data(
            sizing, p.fus,
            wing_llt=llt,
            tail_llt=tail_loading["llt_tail"],
            x_cg_current=p.cg["overall"],
            y_cg=y_cg["overall"],
            Cm_thrust=Cm_front + Cm_back,
            Cm_payload=Cm_payload,
            Vh_V=Vh_V,
        )
    return scissor


def _optimize_battery_y_for_elevator(
    sizing: SizingResult,
    p: _DesignPass,
    *,
    config,
    polar: AirfoilPolar,
    tail_polar: AirfoilPolar,
    airfoil: str,
    llt: LLTResult,
    tail_loading: dict,
    propulsion: PropulsionResult,
    tol: float = 1.0e-5,
) -> tuple[float, dict[str, float], ScissorData, ElevatorResult, float]:
    """Place the battery within the feasible vertical range."""
    elevator_inputs_unchecked = dataclasses.replace(
        config.ELEVATOR,
        enforce_tail_stall=False,
    )
    base_y_cg = weights_mass.compute_y_cg(
        sizing=sizing, fus=p.fus, structure=p.struct,
        masses=p.masses, airfoil_path=airfoil, cg=p.cg,
        rib_inputs=config.RIBS,
    )
    box_y0 = base_y_cg["fuselage"] - 0.5 * p.fus.box_height
    y_lo = base_y_cg["battery_bottom"]
    y_hi = y_lo

    if y_hi < y_lo:
        y_hi = y_lo

    def evaluate(battery_y0: float):
        y_cg = weights_mass.compute_y_cg(
            sizing=sizing, fus=p.fus, structure=p.struct,
            masses=p.masses, airfoil_path=airfoil, cg=p.cg,
            battery_y0=battery_y0,
            rib_inputs=config.RIBS,
        )
        scissor = _final_scissor_for_y_cg(
            sizing, p, config=config, llt=llt,
            tail_loading=tail_loading, y_cg=y_cg,
            propulsion=propulsion,
        )
        elevator_result = elevator.run(
            sizing, scissor, llt, propulsion,
            polar, tail_polar, y_cg, elevator_inputs_unchecked,
            q_sizing=0.5 * sizing.rho * config.V_STALL ** 2,
            payload_max_tension=config.PAYLOAD_MAX_TENSION,
        )
        margin = elevator_result.tail_CL_limit - abs(elevator_result.CLh_cruise)
        return y_cg, scissor, elevator_result, margin

    def checked_elevator(y_cg: dict[str, float], scissor: ScissorData) -> ElevatorResult:
        return elevator.run(
            sizing, scissor, llt, propulsion,
            polar, tail_polar, y_cg, config.ELEVATOR,
            q_sizing=0.5 * sizing.rho * config.V_STALL ** 2,
            payload_max_tension=config.PAYLOAD_MAX_TENSION,
        )

    def checked_candidate(battery_y0: float):
        y_cg_candidate, scissor_candidate, _, margin_candidate = evaluate(battery_y0)
        if margin_candidate < -tol:
            raise ValueError(
                "Battery vertical placement failed: candidate battery height "
                "violates the elevator tail-stall limit.\n"
                f"  candidate battery_y0 = {battery_y0:.4f} m, "
                f"battery_y = {y_cg_candidate['battery']:.4f} m, "
                f"tail |CLh| margin = {margin_candidate:+.4f}"
            )
        elevator_candidate = checked_elevator(y_cg_candidate, scissor_candidate)
        margin_candidate = (
            elevator_candidate.tail_CL_limit - abs(elevator_candidate.CLh_cruise)
        )
        return y_cg_candidate, scissor_candidate, elevator_candidate, margin_candidate

    def cruise_energy_for_candidate(elevator_candidate: ElevatorResult) -> float:
        trim_tail_loading = tail_drag_at_cruise(
            sizing, polar, llt,
            x_cg=p.cg["overall"],
            tail_airfoil=config.TAIL_AIRFOIL,
            CL_tail=elevator_candidate.CLh_cruise,
        )
        cd_full = full_drag_estimate(
            sizing, p.drag, llt,
            cd_i_tail=trim_tail_loading["CD_i_tail_wing_ref"],
        )
        cruise_drag = cd_full * sizing.q_cruise * sizing.Sw
        propulsion_candidate = prop_sizing.run(
            sizing,
            config.PROPULSION,
            cruise_drag=cruise_drag,
        )
        return float(propulsion_candidate.E_cruise)

    battery_y0_override = getattr(config, "BATTERY_Y0", None)
    battery_y_frac = getattr(config, "BATTERY_Y_FRAC", None)
    if battery_y0_override is not None and battery_y_frac is not None:
        raise ValueError("Set only one of battery_y0 or battery_y_frac.")
    if battery_y_frac is not None:
        frac = float(np.clip(battery_y_frac, 0.0, 1.0))
        battery_y0_override = y_lo + frac * (y_hi - y_lo)
    if battery_y0_override is not None:
        y_fixed = float(np.clip(battery_y0_override, y_lo, y_hi))
        y_cg_fixed, scissor_fixed, elevator_fixed, margin_fixed = checked_candidate(y_fixed)
        return y_fixed, y_cg_fixed, scissor_fixed, elevator_fixed, margin_fixed

    battery_y_mode = str(getattr(config, "BATTERY_Y_MODE", "low")).strip().lower()
    if battery_y_mode not in {"low", "middle", "high", "energy"}:
        raise ValueError(
            "battery_y_mode must be one of 'low', 'middle', 'high', or 'energy' "
            f"(got {battery_y_mode!r})."
        )

    if battery_y_mode == "middle":
        y_mid = box_y0 + 0.5 * (p.fus.box_height - p.fus.battery_height)
        y_mid = float(np.clip(y_mid, y_lo, y_hi))
        y_cg_mid, scissor_mid, elevator_mid, margin_mid = checked_candidate(y_mid)
        return y_mid, y_cg_mid, scissor_mid, elevator_mid, margin_mid

    if battery_y_mode == "high":
        y_cg_hi, scissor_hi, elevator_hi, margin_hi = checked_candidate(y_hi)
        return y_hi, y_cg_hi, scissor_hi, elevator_hi, margin_hi

    if battery_y_mode == "energy":
        n_samples = max(2, int(getattr(config, "BATTERY_Y_SAMPLES", 9)))
        candidates = []
        last_error: Exception | None = None
        for y_probe in np.linspace(y_lo, y_hi, n_samples):
            y_probe = float(y_probe)
            try:
                y_cg_probe, scissor_probe, elevator_probe, margin_probe = checked_candidate(y_probe)
            except ValueError as exc:
                last_error = exc
                continue
            energy_probe = cruise_energy_for_candidate(elevator_probe)
            candidates.append((
                energy_probe,
                y_probe,
                y_cg_probe,
                scissor_probe,
                elevator_probe,
                margin_probe,
            ))
        if not candidates:
            if last_error is not None:
                raise last_error
            raise ValueError("Battery vertical placement failed: no feasible energy samples.")
        _, y_best, y_cg_best, scissor_best, elevator_best, margin_best = min(
            candidates,
            key=lambda item: item[0],
        )
        return y_best, y_cg_best, scissor_best, elevator_best, margin_best

    y_cg_lo, scissor_lo, elevator_lo, margin_lo = evaluate(y_lo)
    if margin_lo < -tol:
        raise ValueError(
            "Battery vertical placement failed: even the baseline battery "
            "height violates the elevator tail-stall limit.\n"
            f"  baseline battery_y0 = {y_lo:.4f} m, "
            f"battery_y = {y_cg_lo['battery']:.4f} m, "
            f"tail |CLh| margin = {margin_lo:+.4f}"
        )

    elevator_lo = checked_elevator(y_cg_lo, scissor_lo)
    margin_lo = elevator_lo.tail_CL_limit - abs(elevator_lo.CLh_cruise)
    return y_lo, y_cg_lo, scissor_lo, elevator_lo, margin_lo


def _is_elevator_tail_margin_failure(exc: Exception) -> bool:
    if not isinstance(exc, ValueError):
        return False
    message = str(exc)
    return (
        "violates the elevator tail-stall limit" in message
        or "undeflected cruise tail lift coefficient" in message
        or "trimmed tail is already beyond the allowed finite-tail" in message
    )


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
    s = sizing.inputs
    CL_sw, CD_wing_sw = wing_drag_polar(sizing, polar, alpha_range)
    CD_nonwing = drag.CD0_tail_h + drag.CD0_tail_v + drag.CD0_fus
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
          f"fus LxD={p.fus.length:.3f}x{p.fus.d_eq:.3f}")


def run_pipeline(config) -> PipelineResult:
    """Run the full design pipeline and return a bundled result."""
    # ----- Step 0: resolve airfoil → read t/c from its geometry -----
    airfoil = resolve_airfoil(config.AIRFOIL)
    tc = airfoil_thickness_to_chord(airfoil)
    print(f"Airfoil: {airfoil}  (t/c = {tc:.4f})")
    sizing_inputs = config.SIZING
    # If the user did not supply L_boom in the config, initialize it here.
    # Prefer a conservative default (1.5 m) but clamp to the configured
    # maximum nose→tailback length if provided.
    if getattr(sizing_inputs, "L_boom", None) is None:
        max_len = getattr(sizing_inputs, "max_nose_to_tailback_length", None)
        default_guess = 1.5
        if max_len is not None and max_len > 0.0:
            initial_L = min(default_guess, float(max_len))
        else:
            initial_L = default_guess
        import dataclasses as _dc
        sizing_inputs = _dc.replace(sizing_inputs, L_boom=float(initial_L))

    # ----- Step 1: initial sizing with guessed Cd0 -----
    sizing = wing.run(
        sizing_inputs,
        t_over_c_root=tc,
        c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
    )

    # ----- Step 1b: load airfoil polar once -----
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

    AR = sizing_inputs.AR
    CL_max_wing = Cl_max * AR / (AR + 2.0)
    CL_stall_bound_init = (config.V_STALL / sizing_inputs.V_cruise) ** 2 * CL_max_wing
    print(f"Stall: V_stall = {config.V_STALL:.2f} m/s, "
          f"CL_max_wing = Cl_max · AR/(AR+2) = {CL_max_wing:.3f}  →  "
          f"CL_cruise_bound = (V_stall/V_cruise)² · CL_max_wing = {CL_stall_bound_init:.3f}")
    failure_margin_init = CL_max_wing - sizing.CL_one_drone_failure
    failure_flag_init = "OK" if failure_margin_init >= -1.0e-9 else "FAIL"
    print(f"One-drone-failure lift: CL_req = {sizing.CL_one_drone_failure:.3f}, "
          f"CL_max_wing = {CL_max_wing:.3f}, margin = {failure_margin_init:+.3f}  "
          f"[{failure_flag_init}]")

    # ----- Steps 2-4: convergence loop -----
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
    battery_x = config.BATTERY_X

    cd0_prev = cd0_guess
    m_prev = m_drone_guess
    sw_prev = sw_guess
    dCD0 = dM = dSw = float("inf")
    m_drone = m_drone_guess
    converged = False
    it = 0
    cd_i_tail_prev: float = 0.0

    for it in range(1, config.N_ITER_MAX + 1):
        battery_x, p, scissor_state, cg_error = _optimize_battery_x_for_scissor(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
            battery_x_initial=battery_x,
        )
        # Enforce optional maximum nose-tip → tail-boom-back length from config.SIZING
        max_len = getattr(config.SIZING, "max_nose_to_tailback_length", None)
        if max_len is not None and max_len > 0.0:
            # Compute boom-root x (matches tail_boom_bending_length logic)
            x_aileron_hinge = (
                1.0 - p.control_surface.inputs.c_aileron_to_c_wing
            ) * sizing.c_root
            x_boom_root = (
                sizing.inputs.boom_root_x if sizing.inputs.boom_root_x is not None else x_aileron_hinge
            )
            tail_te_x = float(x_boom_root) + float(sizing.inputs.L_boom)
            nose_tip_x = p.fus.x_nose
            length_nose_to_tail = tail_te_x - nose_tip_x
            # Compute the L_boom that would make nose→tail == max_len:
            desired_L = float(max_len - (x_boom_root - nose_tip_x))
            desired_L = max(0.0, desired_L)
            # If current length is less than the allowed max, increase L_boom
            # to 'max out' the boom (user requested behavior). If current
            # length exceeds max, reduce it as before.
            if length_nose_to_tail < max_len and desired_L > sizing.inputs.L_boom + 1e-9:
                print(
                    f"    Enforcing max nose→tail length: increasing L_boom "
                    f"{sizing.inputs.L_boom:.3f} → {desired_L:.3f} m"
                )
                sizing_inputs = dataclasses.replace(sizing_inputs, L_boom=desired_L)
                sizing = wing.run(
                    sizing_inputs,
                    t_over_c_root=tc,
                    c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
                )
                battery_x, p, scissor_state, cg_error = _optimize_battery_x_for_scissor(
                    sizing,
                    config=config,
                    polar=polar,
                    tail_polar=tail_polar,
                    airfoil=airfoil,
                    battery_x_initial=battery_x,
                )
            elif length_nose_to_tail > max_len:
                # Reduce if it somehow exceeded the max (preserve previous behavior)
                new_L = desired_L
                if new_L < sizing.inputs.L_boom - 1e-9:
                    print(
                        f"    Enforcing max nose→tail length: reducing L_boom "
                        f"{sizing.inputs.L_boom:.3f} → {new_L:.3f} m"
                    )
                    sizing_inputs = dataclasses.replace(sizing_inputs, L_boom=new_L)
                    sizing = wing.run(
                        sizing_inputs,
                        t_over_c_root=tc,
                        c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
                    )
                    battery_x, p, scissor_state, cg_error = _optimize_battery_x_for_scissor(
                        sizing,
                        config=config,
                        polar=polar,
                        tail_polar=tail_polar,
                        airfoil=airfoil,
                        battery_x_initial=battery_x,
                    )
        m_drone = p.masses["total"]
        x_cg = p.cg["overall"]
        scissor = scissor_state.scissor

        Sh_S_new = scissor_state.ShS_target
        Sh_new = Sh_S_new * sizing.Sw

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

        replace_kwargs: dict = {
            "Cd0": p.drag.CD0 + cd_i_tail_prev,
            "AR": AR_new,
            "Sh": Sh_new,
        }

        if config.MASS_CLOSURE:
            replace_kwargs["m_drone_empty"] = m_drone
        sizing_inputs = _sizing_inputs_with_boom_root(
            sizing_inputs,
            battery_x,
            p.fus,
            **replace_kwargs,
        )
        sizing = wing.run(
            sizing_inputs,
            t_over_c_root=tc,
            c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
        )

        cd_i_tail_prev = scissor_state.tail_loading["CD_i_tail_wing_ref"]

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
        print(f"             battery_x={battery_x:.4f}  stability_x@Sh/S={scissor_state.x_target:.4f}  "
              f"x_cg-stability_x={cg_error:+.2e}  cd_i_tail={cd_i_tail_prev:.5f}")

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

    sizing, battery_x, p, final_scissor_state, final_cg_error, final_ShS_error = (
        _retune_tail_area_and_battery_for_scissor(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
            t_over_c_root=tc,
            battery_x_initial=battery_x,
        )
    )
    sizing_inputs = sizing.inputs
    print(
        f"    Scissor coupled tuning: battery_x={battery_x:.4f} m, "
        f"x_cg={p.cg['overall']:.4f} m, stability_x@Sh/S={final_scissor_state.x_target:.4f} m, "
        f"x_error={final_cg_error:+.2e} m, "
        f"Sh/S={sizing.Sh / sizing.Sw:.4f}, scissor_min_Sh/S={final_scissor_state.ShS_target:.4f}, "
        f"Sh/S_error={final_ShS_error:+.2e}"
    )
    battery_x, p, final_scissor_state, final_cg_error = _optimize_battery_x_for_scissor(
        sizing,
        config=config,
        polar=polar,
        tail_polar=tail_polar,
        airfoil=airfoil,
        battery_x_initial=battery_x,
    )
    print(f"    Battery CG tuning: battery_x={battery_x:.4f} m, "
          f"x_cg={p.cg['overall']:.4f} m, stability_x@Sh/S={final_scissor_state.x_target:.4f} m, "
          f"error={final_cg_error:+.2e} m")
    if np.isclose(p.fus.height, config.FUSELAGE.casing_factor * p.fus.battery_height):
        print("    WARNING: fuselage height is just casing_factor × battery_height; "
              "airfoil height is not being used for fuselage sizing.")

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
    failure_lift_achievable = sizing.CL_one_drone_failure <= CL_max_wing
    failure_margin = CL_max_wing - sizing.CL_one_drone_failure
    failure_flag = "OK" if failure_lift_achievable else "FAIL"
    print(f"    One-drone-failure CL check: CL_req = {sizing.CL_one_drone_failure:.3f}, "
          f"CL_max_wing = {CL_max_wing:.3f}, margin = {failure_margin:+.3f}  "
          f"[{failure_flag}]")

    # ----- Step 6: CL/CD sweep -----
    CL_sweep, CD_wing_sweep = wing_drag_polar(sizing, polar, config.ALPHA_SWEEP_DEG)
    CD_nonwing = p.drag.CD0_tail_h + p.drag.CD0_tail_v + p.drag.CD0_fus
    CD_payload = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
        sizing.inputs.n_drones * sizing.Sw
    )
    CD_drone_sweep = CD_wing_sweep + CD_nonwing
    CD_full_sweep = CD_drone_sweep + CD_payload

    # ----- Step 7: tail trim loading + induced drag -----
    CL_tail_trim = (
        1.0e-3 if config.ELEVATOR.trim_mode.lower() == "payload_attachment" else None
    )
    tail_loading = tail_drag_at_cruise(
        sizing, polar, llt,
        x_cg=p.cg["overall"],
        tail_airfoil=config.TAIL_AIRFOIL,
        CL_tail=CL_tail_trim,
    )
    tail_loading_for_scissor = tail_loading
    if config.ELEVATOR.trim_mode.lower() == "payload_attachment":
        tail_loading_for_scissor = tail_drag_at_cruise(
            sizing, polar, llt,
            x_cg=p.cg["overall"],
            tail_airfoil=config.TAIL_AIRFOIL,
            CL_tail=0.5,
        )
    cd_i_tail = tail_loading["CD_i_tail_wing_ref"]
    CD_full_buildup = full_drag_estimate(sizing, p.drag, llt, cd_i_tail=cd_i_tail)

    def refresh_after_tail_area_change(
        sizing_new: SizingResult,
        battery_x_initial: float | None,
        *,
        CL_tail_for_scissor: float | None,
    ):
        battery_x_new, p_new, scissor_state_new, cg_error_new = (
            _optimize_battery_x_for_scissor(
                sizing_new,
                config=config,
                polar=polar,
                tail_polar=tail_polar,
                airfoil=airfoil,
                battery_x_initial=battery_x_initial,
            )
        )
        llt_new = llt_at_cl(sizing_new, polar, CL_target=sizing_new.CL)
        tail_loading_new = tail_drag_at_cruise(
            sizing_new, polar, llt_new,
            x_cg=p_new.cg["overall"],
            tail_airfoil=config.TAIL_AIRFOIL,
            CL_tail=CL_tail_for_scissor,
        )
        cd_i_tail_new = tail_loading_new["CD_i_tail_wing_ref"]
        CD_full_new = full_drag_estimate(
            sizing_new, p_new.drag, llt_new, cd_i_tail=cd_i_tail_new,
        )
        cruise_drag_new = CD_full_new * sizing_new.q_cruise * sizing_new.Sw
        propulsion_new = prop_sizing.run(
            sizing_new,
            config.PROPULSION,
            cruise_drag=cruise_drag_new,
        )
        return (
            sizing_new, battery_x_new, p_new, scissor_state_new, cg_error_new,
            llt_new, tail_loading_new, cd_i_tail_new, CD_full_new, propulsion_new,
        )

    def increase_tail_area_for_margin(factor: float = 1.10, *, label: str = "Elevator"):
        nonlocal sizing, sizing_inputs, battery_x, p, final_scissor_state
        nonlocal final_cg_error, llt, tail_loading, tail_loading_for_scissor
        nonlocal cd_i_tail, CD_full_buildup, final_propulsion

        Sh_old = sizing.Sh
        Sw_old = sizing.Sw
        sizing_inputs = _sizing_inputs_with_boom_root(
            sizing.inputs,
            battery_x,
            p.fus,
            Sh=factor * sizing.Sh,
        )
        sizing = wing.run(
            sizing_inputs,
            t_over_c_root=tc,
            c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
        )
        sizing_inputs = sizing.inputs
        (
            sizing, battery_x, p, final_scissor_state, final_cg_error,
            llt, tail_loading, cd_i_tail, CD_full_buildup, final_propulsion,
        ) = refresh_after_tail_area_change(
            sizing,
            battery_x,
            CL_tail_for_scissor=CL_tail_trim,
        )
        tail_loading_for_scissor = tail_loading
        if config.ELEVATOR.trim_mode.lower() == "payload_attachment":
            tail_loading_for_scissor = tail_drag_at_cruise(
                sizing, polar, llt,
                x_cg=p.cg["overall"],
                tail_airfoil=config.TAIL_AIRFOIL,
                CL_tail=0.5,
            )
        print(
            f"    {label} tail CL margin tuning: increased Sh/S "
            f"from {Sh_old / Sw_old:.4f} to {sizing.Sh / sizing.Sw:.4f}"
        )

    def set_tail_area_to_ShS(
        ShS_target: float,
        *,
        battery_x_for_area: float | None = None,
        fus_for_area: FuselageResult | None = None,
    ):
        nonlocal sizing, sizing_inputs

        area_battery_x = battery_x if battery_x_for_area is None else battery_x_for_area
        area_fus = p.fus if fus_for_area is None else fus_for_area
        sizing_inputs = _sizing_inputs_with_boom_root(
            sizing.inputs,
            area_battery_x,
            area_fus,
            Sh=ShS_target * sizing.Sw,
        )
        sizing = wing.run(
            sizing_inputs,
            t_over_c_root=tc,
            c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
        )
        sizing_inputs = sizing.inputs

    # ----- Step 8: vertical battery placement + final scissor/elevator -----
    final_propulsion = p.propulsion
    for tail_area_iter in range(10):
        try:
            battery_y0, y_cg, scissor, elevator_result, elevator_cl_margin = (
                _optimize_battery_y_for_elevator(
                    sizing, p,
                    config=config,
                    polar=polar,
                    tail_polar=tail_polar,
                    airfoil=airfoil,
                    llt=llt,
                    tail_loading=tail_loading_for_scissor,
                    propulsion=final_propulsion,
                )
            )
            break
        except ValueError as exc:
            if not _is_elevator_tail_margin_failure(exc) or tail_area_iter == 9:
                raise
            increase_tail_area_for_margin(1.15, label="Elevator")
    print(f"    Battery vertical placement: battery_y0={battery_y0:.4f} m, "
          f"battery_y={y_cg['battery']:.4f} m, y_cg={y_cg['overall']:.4f} m, "
          f"tail |CLh| margin={elevator_cl_margin:+.4f}")

    # ----- Step 9: rudder sizing -----
    rudder_result = rudder.run(
        sizing, scissor, p.fus, config.V_STALL, config.RUDDER,
        x_cg=p.cg["overall"],
        total_thrust=final_propulsion.thrust_cruise_per_prop * final_propulsion.inputs.n_props,
        y_cg=y_cg,
        payload_max_tension=config.PAYLOAD_MAX_TENSION,
    )

    # ----- Step 7b: recompute tail drag at trimmed CL_tail -----
    for _ in range(3):
        tail_loading = tail_drag_at_cruise(
            sizing, polar, llt,
            x_cg=p.cg["overall"],
            tail_airfoil=config.TAIL_AIRFOIL,
            CL_tail=elevator_result.CLh_cruise,
        )
        tail_loading_for_scissor = tail_loading
        if config.ELEVATOR.trim_mode.lower() == "payload_attachment":
            tail_loading_for_scissor = tail_drag_at_cruise(
                sizing, polar, llt,
                x_cg=p.cg["overall"],
                tail_airfoil=config.TAIL_AIRFOIL,
                CL_tail=0.5,
            )
        cd_i_tail = tail_loading["CD_i_tail_wing_ref"]
        CD_full_buildup = full_drag_estimate(sizing, p.drag, llt, cd_i_tail=cd_i_tail)
        cruise_drag = CD_full_buildup * sizing.q_cruise * sizing.Sw
        updated_propulsion = prop_sizing.run(
            sizing,
            config.PROPULSION,
            cruise_drag=cruise_drag,
        )
        if abs(updated_propulsion.thrust_cruise_per_prop - final_propulsion.thrust_cruise_per_prop) < 1.0e-3:
            final_propulsion = updated_propulsion
            break
        final_propulsion = updated_propulsion
        for tail_area_iter in range(6):
            try:
                battery_y0, y_cg, scissor, elevator_result, elevator_cl_margin = (
                    _optimize_battery_y_for_elevator(
                        sizing, p,
                        config=config,
                        polar=polar,
                        tail_polar=tail_polar,
                        airfoil=airfoil,
                        llt=llt,
                        tail_loading=tail_loading_for_scissor,
                        propulsion=final_propulsion,
                    )
                )
                break
            except ValueError as exc:
                if not _is_elevator_tail_margin_failure(exc) or tail_area_iter == 5:
                    raise
                increase_tail_area_for_margin(1.10, label="Propulsion refresh elevator")
    tail_loading = tail_drag_at_cruise(
        sizing, polar, llt,
        x_cg=p.cg["overall"],
        tail_airfoil=config.TAIL_AIRFOIL,
        CL_tail=elevator_result.CLh_cruise,
    )
    cd_i_tail = tail_loading["CD_i_tail_wing_ref"]
    CD_full_buildup = full_drag_estimate(sizing, p.drag, llt, cd_i_tail=cd_i_tail)
    cruise_drag = CD_full_buildup * sizing.q_cruise * sizing.Sw
    final_propulsion = prop_sizing.run(
        sizing,
        config.PROPULSION,
        cruise_drag=cruise_drag,
    )
    for tail_area_iter in range(6):
        try:
            battery_y0, y_cg, scissor, elevator_result, elevator_cl_margin = (
                _optimize_battery_y_for_elevator(
                    sizing, p,
                    config=config,
                    polar=polar,
                    tail_polar=tail_polar,
                    airfoil=airfoil,
                    llt=llt,
                    tail_loading=tail_loading,
                    propulsion=final_propulsion,
                )
            )
            break
        except ValueError as exc:
            if not _is_elevator_tail_margin_failure(exc) or tail_area_iter == 5:
                raise
            increase_tail_area_for_margin(1.10, label="Final elevator")
            tail_loading = tail_loading_for_scissor

    def evaluate_final_scissor(candidate_battery_x: float):
        p_eval = _run_design_pass(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
            battery_x=candidate_battery_x,
        )
        llt_eval = llt_at_cl(sizing, polar, CL_target=sizing.CL)
        tail_loading_eval = tail_drag_at_cruise(
            sizing, polar, llt_eval,
            x_cg=p_eval.cg["overall"],
            tail_airfoil=config.TAIL_AIRFOIL,
            CL_tail=elevator_result.CLh_cruise,
        )
        cd_i_tail_eval = tail_loading_eval["CD_i_tail_wing_ref"]
        CD_full_eval = full_drag_estimate(
            sizing, p_eval.drag, llt_eval, cd_i_tail=cd_i_tail_eval,
        )
        cruise_drag_eval = CD_full_eval * sizing.q_cruise * sizing.Sw
        propulsion_eval = prop_sizing.run(
            sizing,
            config.PROPULSION,
            cruise_drag=cruise_drag_eval,
        )
        battery_y0_eval, y_cg_eval, scissor_eval, elevator_eval, margin_eval = (
            _optimize_battery_y_for_elevator(
                sizing, p_eval,
                config=config,
                polar=polar,
                tail_polar=tail_polar,
                airfoil=airfoil,
                llt=llt_eval,
                tail_loading=tail_loading_eval,
                propulsion=propulsion_eval,
            )
        )
        ShS_current_eval = sizing.Sh / sizing.Sw
        x_target_eval, _ = _stability_x_at_ShS(
            scissor_eval,
            ShS_current_eval,
            reference_x=p_eval.cg["overall"],
        )
        _, ShS_target_eval = _scissor_intersection(scissor_eval)
        return {
            "battery_x": float(candidate_battery_x),
            "p": p_eval,
            "llt": llt_eval,
            "tail_loading": tail_loading_eval,
            "cd_i_tail": cd_i_tail_eval,
            "CD_full_buildup": CD_full_eval,
            "propulsion": propulsion_eval,
            "battery_y0": battery_y0_eval,
            "y_cg": y_cg_eval,
            "scissor": scissor_eval,
            "elevator": elevator_eval,
            "margin": margin_eval,
            "x_target": x_target_eval,
            "ShS_target": ShS_target_eval,
            "x_error": p_eval.cg["overall"] - x_target_eval,
            "ShS_error": sizing.Sh / sizing.Sw - ShS_target_eval,
        }

    def evaluate_final_scissor_at_stability(candidate_battery_x: float | None):
        battery_x_eval, _, _, _ = _optimize_battery_x_for_scissor(
            sizing,
            config=config,
            polar=polar,
            tail_polar=tail_polar,
            airfoil=airfoil,
            battery_x_initial=candidate_battery_x,
            max_iter=3,
        )
        return evaluate_final_scissor(float(battery_x_eval))

    def relax_tail_area_if_margin_inactive(final_state: dict) -> dict:
        if (
            final_state["ShS_error"] <= 1.0e-4
            or final_state["margin"] <= 1.0e-4
        ):
            return final_state

        old_ShS = sizing.Sh / sizing.Sw
        old_state = final_state
        elevator_eval = old_state["elevator"]
        tail_limit = elevator_eval.tail_CL_limit
        abs_CLh = abs(elevator_eval.CLh_cruise)
        if tail_limit <= 0.0 or abs_CLh <= 0.0:
            return final_state

        stall_ShS = old_ShS * abs_CLh / tail_limit
        target_ShS = max(final_state["ShS_target"], stall_ShS)
        if target_ShS >= old_ShS - 1.0e-4:
            return final_state

        set_tail_area_to_ShS(
            target_ShS,
            battery_x_for_area=old_state["battery_x"],
            fus_for_area=old_state["p"].fus,
        )
        try:
            target_state = evaluate_final_scissor_at_stability(old_state["battery_x"])
        except ValueError as exc:
            if not _is_elevator_tail_margin_failure(exc):
                raise
            set_tail_area_to_ShS(
                old_ShS,
                battery_x_for_area=old_state["battery_x"],
                fus_for_area=old_state["p"].fus,
            )
            return old_state
        if target_state["margin"] < -1.0e-4 or target_state["x_error"] > 1.0e-4:
            lo_ShS = target_ShS
            hi_ShS = old_ShS
            best_state = old_state
            best_ShS = old_ShS
            for _ in range(4):
                mid_ShS = 0.5 * (lo_ShS + hi_ShS)
                set_tail_area_to_ShS(
                    mid_ShS,
                    battery_x_for_area=old_state["battery_x"],
                    fus_for_area=old_state["p"].fus,
                )
                try:
                    mid_state = evaluate_final_scissor_at_stability(old_state["battery_x"])
                except ValueError as exc:
                    if not _is_elevator_tail_margin_failure(exc):
                        raise
                    lo_ShS = mid_ShS
                    continue
                if mid_state["margin"] >= -1.0e-4 and mid_state["x_error"] <= 1.0e-4:
                    best_state = mid_state
                    best_ShS = mid_ShS
                    hi_ShS = mid_ShS
                else:
                    lo_ShS = mid_ShS
            if best_state is not old_state:
                set_tail_area_to_ShS(
                    best_ShS,
                    battery_x_for_area=best_state["battery_x"],
                    fus_for_area=best_state["p"].fus,
                )
                print(
                    f"    Final elevator tail CL margin active: reduced Sh/S "
                    f"from {old_ShS:.4f} to {sizing.Sh / sizing.Sw:.4f}"
                )
                return best_state
            set_tail_area_to_ShS(
                old_ShS,
                battery_x_for_area=old_state["battery_x"],
                fus_for_area=old_state["p"].fus,
            )
            return old_state

        label = "inactive" if target_state["ShS_error"] <= 1.0e-4 else "active"
        print(
            f"    Final elevator tail CL margin {label}: reduced Sh/S "
            f"from {old_ShS:.4f} to {sizing.Sh / sizing.Sw:.4f}"
        )
        return target_state

    def relax_tail_area_until_active_or_scissor(final_state: dict) -> dict:
        for _ in range(2):
            old_ShS = sizing.Sh / sizing.Sw
            relaxed_state = relax_tail_area_if_margin_inactive(final_state)
            if relaxed_state is final_state:
                return final_state
            final_state = relaxed_state
            if abs(sizing.Sh / sizing.Sw - old_ShS) <= 1.0e-4:
                return final_state
        return final_state

    for tail_area_iter in range(10):
        try:
            final_state = evaluate_final_scissor(float(battery_x))
            break
        except ValueError as exc:
            if not _is_elevator_tail_margin_failure(exc) or tail_area_iter == 9:
                raise
            increase_tail_area_for_margin(1.10, label="Final elevator")
    for _ in range(6):
        desired_boom_root_x = _boom_root_x_from_battery(
            final_state["battery_x"],
            final_state["p"].fus,
        )
        if desired_boom_root_x is not None and not np.isclose(
            sizing.inputs.boom_root_x,
            desired_boom_root_x,
            atol=1.0e-5,
            rtol=0.0,
        ):
            sizing_inputs = dataclasses.replace(
                sizing.inputs,
                boom_root_x=desired_boom_root_x,
            )
            sizing = wing.run(
                sizing_inputs,
                t_over_c_root=tc,
                c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
            )
            sizing_inputs = sizing.inputs
            try:
                final_state = evaluate_final_scissor(final_state["battery_x"])
            except ValueError as exc:
                if not _is_elevator_tail_margin_failure(exc):
                    raise
                increase_tail_area_for_margin(1.01, label="Final elevator")
                final_state = evaluate_final_scissor(final_state["battery_x"])
            continue

        if (
            abs(final_state["x_error"]) <= 1.0e-4
            and (
                abs(final_state["ShS_error"]) <= 1.0e-4
                or final_state["margin"] <= 1.0e-4
            )
            and final_state["margin"] >= -1.0e-4
        ):
            break

        if final_state["margin"] < -1.0e-4:
            increase_tail_area_for_margin(1.10, label="Final elevator")
            final_state = evaluate_final_scissor(final_state["battery_x"])
            continue

        if final_state["ShS_error"] < -1.0e-4:
            sizing_inputs = _sizing_inputs_with_boom_root(
                sizing.inputs,
                final_state["battery_x"],
                final_state["p"].fus,
                Sh=final_state["ShS_target"] * sizing.Sw,
            )
            sizing = wing.run(
                sizing_inputs,
                t_over_c_root=tc,
                c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
            )
            sizing_inputs = sizing.inputs
            try:
                final_state = evaluate_final_scissor(final_state["battery_x"])
            except ValueError as exc:
                if not _is_elevator_tail_margin_failure(exc):
                    raise
                increase_tail_area_for_margin(1.01, label="Final elevator")
                final_state = evaluate_final_scissor(final_state["battery_x"])
            continue

        prev_state = final_state
        probe_step = max(0.05 * sizing.c, 0.01)
        probe_state = evaluate_final_scissor(final_state["battery_x"] + probe_step)
        best_state = min(
            (prev_state, probe_state),
            key=lambda st: abs(st["x_error"]),
        )
        for _ in range(6):
            dx = probe_state["battery_x"] - prev_state["battery_x"]
            de = probe_state["x_error"] - prev_state["x_error"]
            if abs(de) < 1.0e-9 or abs(dx) < 1.0e-9:
                next_x = probe_state["battery_x"] - np.sign(probe_state["x_error"]) * probe_step
            else:
                next_x = probe_state["battery_x"] - probe_state["x_error"] * dx / de
            max_step = max(0.35 * sizing.c, 0.05)
            next_x = float(np.clip(
                next_x,
                probe_state["battery_x"] - max_step,
                probe_state["battery_x"] + max_step,
            ))
            next_state = evaluate_final_scissor(next_x)
            if abs(next_state["x_error"]) < abs(best_state["x_error"]):
                best_state = next_state
            if abs(next_state["x_error"]) <= 1.0e-4:
                best_state = next_state
                break
            prev_state, probe_state = probe_state, next_state
        final_state = best_state

    for _ in range(6):
        rudder_area_req = rudder.minimum_vertical_tail_area(
            sizing,
            final_state["p"].fus,
            config.V_STALL,
            config.RUDDER,
            x_cg=final_state["p"].cg["overall"],
            total_thrust=(
                final_state["propulsion"].thrust_cruise_per_prop
                * final_state["propulsion"].inputs.n_props
            ),
            y_cg=final_state["y_cg"],
            payload_max_tension=config.PAYLOAD_MAX_TENSION,
        )
        sv_old = sizing.Sv
        if abs(rudder_area_req.Sv - sv_old) <= max(1.0e-6, 1.0e-5 * sv_old):
            print(
                f"    Rudder VT area check: current Sv={sv_old:.4f} m^2, "
                f"required Sv={rudder_area_req.Sv:.4f} m^2, "
                f"required bR/bV={rudder_area_req.bR_bV_required:.4f}"
            )
            break
        Vv_new = rudder_area_req.Sv * sizing.lh / (sizing.Sw * sizing.inputs.b)
        sizing_inputs = dataclasses.replace(sizing.inputs, Vv=Vv_new)
        sizing = wing.run(
            sizing_inputs,
            t_over_c_root=tc,
            c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
        )
        final_state = evaluate_final_scissor(final_state["battery_x"])
        print(
            f"    Rudder VT area tuning: Sv {sv_old:.4f} -> {sizing.Sv:.4f} m^2 "
            f"(Vv={sizing.inputs.Vv:.4f}, required bR/bV={rudder_area_req.bR_bV_required:.4f})"
        )

    final_state = relax_tail_area_until_active_or_scissor(final_state)

    battery_x = final_state["battery_x"]
    p = final_state["p"]
    llt = final_state["llt"]
    tail_loading = final_state["tail_loading"]
    cd_i_tail = final_state["cd_i_tail"]
    CD_full_buildup = final_state["CD_full_buildup"]
    final_propulsion = final_state["propulsion"]
    battery_y0 = final_state["battery_y0"]
    y_cg = final_state["y_cg"]
    scissor = final_state["scissor"]
    elevator_result = final_state["elevator"]
    elevator_cl_margin = final_state["margin"]

    if mass_history:
        mass_history[-1] = p.masses["total"]
    if cg_history:
        cg_history[-1] = p.cg["overall"]

    final_scissor_x_target, _ = _stability_x_at_ShS(
        scissor,
        sizing.Sh / sizing.Sw,
        reference_x=p.cg["overall"],
    )
    _, final_scissor_ShS_target = _scissor_intersection(scissor)
    print(
        f"    Final scissor closure: x_cg={p.cg['overall']:.4f} m, "
        f"stability_x@Sh/S={final_scissor_x_target:.4f} m, "
        f"x_error={p.cg['overall'] - final_scissor_x_target:+.2e} m, "
        f"Sh/S={sizing.Sh / sizing.Sw:.4f}, "
        f"scissor_min_Sh/S={final_scissor_ShS_target:.4f}, "
        f"Sh/S_error={sizing.Sh / sizing.Sw - final_scissor_ShS_target:+.2e}"
    )

    rudder_result = rudder.run(
        sizing, scissor, p.fus, config.V_STALL, config.RUDDER,
        x_cg=p.cg["overall"],
        total_thrust=final_propulsion.thrust_cruise_per_prop * final_propulsion.inputs.n_props,
        y_cg=y_cg,
        payload_max_tension=config.PAYLOAD_MAX_TENSION,
    )

    # ----- Step 10: torsion check + physics-based VT rod sizing -----
    vt_geom = WingGeometry(b=sizing.bv, S=sizing.Sv, taper=sizing.inputs.lam_t)
    vt_flight = FlightCondition(
        V_inf=sizing.inputs.V_cruise, rho=sizing.rho, CL_target=0.5,
    )
    vt_llt = solve_llt(vt_geom, tail_polar, vt_flight)
    CLalphav_vt = vt_llt.CL / (vt_llt.alpha_root - tail_polar.alpha_L0)

    struct = rods.run(
        sizing,
        p.control_surface,
        final_propulsion,
        airfoil,
        config.TAIL_AIRFOIL,
        config.STRUCTURE,
        tail_polar=tail_polar,
        rudder=rudder_result,
        CLalphav_vt=CLalphav_vt,
    )

    struct = rods.apply_torsion_check(
        rod=struct,
        rudder_hinge_moment=rudder_result.hinge_moment.H,
        bending_force=struct.F_tail_structural,
        boom_length=rods.tail_boom_bending_length(sizing, p.control_surface),
        tube_length=sizing.L_boom,
        safety_factor=config.STRUCTURE.safety_factor,
    )

    final_masses = weights_mass.total_mass(
        sizing=sizing,
        propulsion=final_propulsion,
        structure=struct,
        fus=p.fus,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
        aileron=p.control_surface,
        battery_x=battery_x,
        materials=config.MATERIALS,
        rib_inputs=config.RIBS,
        rudder_inputs=config.RUDDER,
    )
    final_cg = weights_mass.compute_cg(
        sizing=sizing,
        propulsion=final_propulsion,
        structure=struct,
        fus=p.fus,
        aileron=p.control_surface,
        airfoil_path=airfoil,
        tail_airfoil_path=config.TAIL_AIRFOIL,
        battery_x=battery_x,
        materials=config.MATERIALS,
        rib_inputs=config.RIBS,
        rudder_inputs=config.RUDDER,
    )
    final_y_cg = weights_mass.compute_y_cg(
        sizing=sizing,
        fus=p.fus,
        structure=struct,
        masses=final_masses,
        airfoil_path=airfoil,
        cg=final_cg,
        battery_y0=battery_y0,
        rib_inputs=config.RIBS,
    )
    final_rib_checks = weights_mass.rib_checks(
        sizing=sizing,
        propulsion=final_propulsion,
        airfoil_path=airfoil,
        rib_inputs=config.RIBS,
        materials=config.MATERIALS,
    )

    return PipelineResult(
        airfoil=airfoil,
        tail_airfoil=config.TAIL_AIRFOIL,
        tc=tc,
        polar=polar,
        cl_max=Cl_max,
        cl_max_wing=CL_max_wing,
        sizing=sizing,
        propulsion=final_propulsion,
        fus=p.fus,
        drag=p.drag,
        struct=struct,
        control_surface=p.control_surface,
        elevator=elevator_result,
        rudder=rudder_result,
        masses=final_masses,
        cg=final_cg,
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
        failure_lift_achievable=failure_lift_achievable,
        cl_sweep=CL_sweep,
        cd_drone_sweep=CD_drone_sweep,
        cd_full_sweep=CD_full_sweep,
        cd_full_buildup=CD_full_buildup,
        cd_payload=CD_payload,
        cd_i_tail=cd_i_tail,
        tail_loading=tail_loading,
        scissor=scissor,
        rib_checks=final_rib_checks,
        battery_y0=battery_y0,
        y_cg=final_y_cg,
    )

