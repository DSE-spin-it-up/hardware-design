"""Console report blocks for a converged PipelineResult.

`print_main_summary` prints everything from "INITIAL SIZING" through the
α-sweep summary. `print_final_drag` prints the trailing 5-line
full-buildup CD block. The split matches the historical output order
(plot_drone_ld is shown by the caller between the two).
"""
from __future__ import annotations

import numpy as np

from aerodynamics import drag_buildup
from pipeline.loop import PipelineResult
from propulsion import sizing as prop_sizing
from sizing import aileron, elevator, fuselage, rudder, wing
from structures import rods
from weights.inertia import calculate_mass_moment_of_inertia
from weights.mass import epp_foam_volumes


def _ok_marker(value: bool) -> str:
    return "OK" if value else "FAIL"


def _print_rib_checks(result: PipelineResult) -> None:
    if not result.rib_checks:
        return
    print("\n--- Wing ribs ---")
    print(f"  Material thickness     : {result.rib_checks[0]['thickness'] * 1e3:.2f}  mm")
    print(f"  Mass (total)           : {result.masses.get('ribs', 0.0):.3f}  kg")
    for check in result.rib_checks:
        label = str(check["name"]).replace("_", " ").title()
        print(f"  {label} ribs ({int(check['count'])}x):")
        print(
            f"    Minimum rectangle    : {check['bbox_width'] * 1e3:.2f} x "
            f"{check['bbox_height'] * 1e3:.2f}  mm"
        )
        print(f"    Rib thickness        : {check['thickness'] * 1e3:.2f}  mm")
        print(
            f"    Thrust tension       : {check['tension_stress'] / 1e6:.2f} / "
            f"{check['tension_allowable'] / 1e6:.2f} MPa  "
            f"{_ok_marker(bool(check['tension_ok']))}"
        )
        print(
            f"    Lift bending         : {check['bending_stress'] / 1e6:.2f} / "
            f"{check['bending_allowable'] / 1e6:.2f} MPa  "
            f"{_ok_marker(bool(check['bending_ok']))}"
        )
        print(
            f"    Combined von Mises   : {check['combined_stress'] / 1e6:.2f} / "
            f"{check['combined_allowable'] / 1e6:.2f} MPa  "
            f"{_ok_marker(bool(check['combined_ok']))}"
        )
        print(
            f"    Lift moment / Ixx    : {check['lift_moment']:.2f} N*m / "
            f"{check['side_ixx']:.3e} m^4"
        )


def _print_control_authority(result: PipelineResult) -> None:
    a = result.control_surface
    e = result.elevator
    r = result.rudder

    da_max = np.radians(a.inputs.max_da_deg)
    de_up = -np.radians(e.inputs.max_deflection_up_deg)
    de_down = np.radians(e.inputs.max_deflection_down_deg)
    dr_max = r.delta_R

    cl_aileron = a.cl_da * da_max
    cm_elevator_up = e.CM_deltaE * de_up
    cm_elevator_down = e.CM_deltaE * de_down
    cn_rudder = r.Cndr * dr_max

    print("\n----- Maximum Control Moment Coefficients -----")
    print(f"  Aileron max roll  Cl   : {cl_aileron:+.5f}  "
          f"(Cl_delta_a {a.cl_da:+.5f}/rad * {a.inputs.max_da_deg:.1f} deg)")
    print(f"  Elevator max pitch Cm  : {max(abs(cm_elevator_up), abs(cm_elevator_down)):.5f}  magnitude")
    print(f"    up   Cm              : {cm_elevator_up:+.5f}  "
          f"(Cm_delta_e {e.CM_deltaE:+.5f}/rad * {-e.inputs.max_deflection_up_deg:.1f} deg)")
    print(f"    down Cm              : {cm_elevator_down:+.5f}  "
          f"(Cm_delta_e {e.CM_deltaE:+.5f}/rad * {e.inputs.max_deflection_down_deg:.1f} deg)")
    print(f"  Rudder max yaw    Cn   : {cn_rudder:+.5f}  "
          f"(Cn_delta_r {r.Cndr:+.5f}/rad * {np.degrees(dr_max):.1f} deg)")


def _fmt_mass(label: str, value: float, width: int = 28) -> str:
    return f"  {label:<{width}}: {value:.4f}  kg"


def _fmt_area(label: str, value: float, width: int = 28) -> str:
    return f"  {label:<{width}}: {value:.4f}  m^2"


def _fmt_cg(label: str, value: float, width: int = 28) -> str:
    return f"  {label:<{width}}: {value:+8.4f}  m"


def _print_bounding_boxes(result: PipelineResult) -> None:
    s = result.sizing
    f = result.fus

    boxes = [
        (
            "Fuselage",
            f.length,
            f.width,
            f.height,
            1,
            "aero body",
        ),
        (
            "Half wing",
            s.inputs.b / 2.0,
            s.c_root,
            s.t_over_c_root * s.c_root,
            2,
            "each side",
        ),
        (
            "Half horizontal tail",
            s.bh / 2.0,
            s.ch,
            s.tt_h,
            2,
            "each side",
        ),
        (
            "Vertical tail",
            s.cv,
            s.tt_v,
            s.bv,
            1,
            "single fin",
        ),
    ]

    total_volume = 0.0
    print("\n========== PACKAGING / BOUNDING BOXES ==========")
    print("  Dimensions are box L x W x H envelopes for the final geometry.")
    for name, length, width, height, count, note in boxes:
        volume_each = length * width * height
        volume_total = count * volume_each
        total_volume += volume_total
        count_label = f"{count}x " if count > 1 else ""
        print(f"  {count_label}{name:<22}: {length:.4f} x {width:.4f} x {height:.4f}  m  ({note})")
        print(
            f"    Volume each          : {volume_each:.6f}  m^3  "
            f"({volume_each * 1000:.2f} L)"
        )
        if count > 1:
            print(
                f"    Volume total         : {volume_total:.6f}  m^3  "
                f"({volume_total * 1000:.2f} L)"
            )
    print(
        f"  {'Total box volume':<26}: {total_volume:.6f}  m^3  "
        f"({total_volume * 1000:.2f} L)"
    )


def _print_vehicle_block(result: PipelineResult) -> None:
    s = result.sizing
    p = result.propulsion
    cl_alpha = result.scissor.CL_alpha_w
    cl0 = -cl_alpha * result.polar.alpha_L0
    t_max_total = p.max_thrust_per_prop * p.inputs.n_props

    print("\nvehicle:")
    print(f"  m:           {result.masses['total']:.3f}")
    print("  g:           9.81")
    print(f"  rho:         {s.rho:.4f}")
    print(f"  S:           {s.Sw:.4f}")
    print(f"  CL0:         {cl0:.4g}")
    print(f"  AR:          {s.inputs.AR:.4g}")
    print(f"  e:           {s.e:.4f}")
    print(f"  CLa:         {cl_alpha:.4g}")
    print(f"  CD0:         {result.drag.CD0:.5f}")
    print(f"  CD0_payload: {s.inputs.Cd_payload:.4g}")
    print(f"  S_payload:   {s.inputs.S_payload:.4g}")
    print(f"  m_L:         {s.inputs.m_payload:.4g}")
    print("  cable_len:   18.0")
    print("  cable_tol:   0.1")

    print("\nlimits:")
    print("  V_min:         14.0")
    print(f"  V_max:         {s.inputs.v_max:.4g}")
    print(f"  gam_max:       {np.radians(45.0):.6g}")
    print("  T_min:         0.0")
    print(f"  T_max:         {t_max_total:.4g}")
    print(f"  P_max:         {p.max_power_elec:.4g}")
    print(f"  alpha_min:     {np.radians(-15.0):.6g}")
    print(f"  alpha_max:     {np.radians(8.0):.6g}")
    print(f"  mu_max:        {np.radians(35.0):.6g}")
    print("  d_min:         6.0")
    print(f"  V_cruise:      {s.inputs.V_cruise:.4g}")
    print("  Tc_max:        750.0")


def print_main_summary(result: PipelineResult) -> None:
    sizing = result.sizing
    drag = result.drag
    masses = result.masses
    cg = result.cg

    print("========== INITIAL SIZING ==========")
    wing.summary(sizing)
    print("\n========== PROPULSION SYSTEM ==========")
    prop_sizing.summary(result.propulsion)
    print("\n========== FUSELAGE ==========")
    fuselage.summary(result.fus)
    _print_bounding_boxes(result)
    print("\n========== CONTROL SURFACES ==========")
    print("Aileron:")
    aileron.summary(result.control_surface)
    print("Elevator:")
    elevator.summary(result.elevator)
    print("Rudder:")
    rudder.summary(result.rudder)
    _print_control_authority(result)
    print("\n========== STRUCTURE ==========")
    rods.summary(result.struct)
    _print_rib_checks(result)

    # ------------------------------------------------------------------
    # Mass breakdown
    # ------------------------------------------------------------------
    print("\n========== MASS ESTIMATES ==========")
    print("  --- Propulsion ---")
    print(_fmt_mass("Battery",              masses['battery']))
    print(_fmt_mass("Motors (total)",       masses['motors']))
    print(_fmt_mass("Propellers (total)",   masses['props']))

    print("  --- Airframe ---")
    print(_fmt_mass("Wing (foam)",          masses['wing']))
    print(_fmt_mass("Wing ribs",            masses.get('ribs', 0.0)))
    print(_fmt_mass("Horizontal tail",      masses['hor_tail']))
    print(_fmt_mass("Vertical tail",        masses['ver_tail']))
    print(_fmt_mass("Fuselage (shell)",     masses['fuselage']))
    print(_fmt_mass("Structural sleeve",    masses.get('pvc_tubes', 0.0)))

    foam_volumes = epp_foam_volumes(
        sizing=result.sizing,
        fus=result.fus,
        structure=result.struct,
        airfoil_path=result.airfoil,
        tail_airfoil_path=result.tail_airfoil,
        aileron=result.control_surface,
        battery_x=result.cg["battery"],
        materials=result.materials,
    )
    total_foam_volume = foam_volumes["total"]
    print(
        f"  {'Total EPP foam volume':<28}: {total_foam_volume:.6f}  m^3  "
        f"({total_foam_volume * 1000:.2f} L)"
    )

    print("  --- Rods ---")
    print(_fmt_mass("Wing spar rod",        masses['rod_spar']))
    print(_fmt_mass("Wing aileron rod",     masses['rod_aileron']))
    print(_fmt_mass("Tail (boom) rod",      masses['tail_rod']))
    print(_fmt_mass("HT spar rod",          masses.get('ht_spar', 0.0)))
    print(_fmt_mass("HT elevator rod",      masses.get('ht_rud', 0.0)))
    print(_fmt_mass("VT spar rod",          masses.get('vt_spar', 0.0)))
    print(_fmt_mass("VT rudder rod",        masses.get('vt_rud', 0.0)))
    print(_fmt_mass("Rod connectors (3x)",  masses.get('rod_connectors', 0.0)))

    print("  --- Glass sheet ---")
    print(_fmt_mass("Glass sheet (wing)",   masses.get('glass_sheet_wing',   0.0)))
    print(_fmt_mass("Glass sheet (HT)",     masses.get('glass_sheet_tail_h', 0.0)))
    print(_fmt_mass("Glass sheet (VT)",     masses.get('glass_sheet_tail_v', 0.0)))
    print(_fmt_mass("Glass sheet (total)",  masses.get('glass_sheet',        0.0)))
    wing_sheet_area = 2.0 * sizing.Sw
    hor_tail_sheet_area = 2.0 * sizing.Sh
    ver_tail_sheet_area = 2.0 * sizing.Sv
    print(f"  {'GFRP thickness':<28}: {sizing.inputs.glass_sheet_thickness_mm:.3f}  mm")
    print(_fmt_area("GFRP surface area (wing)", wing_sheet_area))
    print(_fmt_area("GFRP surface area (HT)", hor_tail_sheet_area))
    print(_fmt_area("GFRP surface area (VT)", ver_tail_sheet_area))
    print(_fmt_area(
        "GFRP surface area (total)",
        wing_sheet_area + hor_tail_sheet_area + ver_tail_sheet_area,
    ))

    print("  --- Other ---")
    print(_fmt_mass("Servos (total)",       masses.get('servos', 0.0)))
    print(_fmt_mass("Sensors",              masses.get('sensors', 0.0)))
    print(_fmt_mass("Wiring (total)",       masses.get('wiring',  0.0)))

    print(f"\n  {'TOTAL (with margin)':<28}: {masses['total']:.4f}  kg")

    # ------------------------------------------------------------------
    # x-CG breakdown
    # ------------------------------------------------------------------
    print("\n========== CENTER OF GRAVITY (x, from LEMAC) ==========")
    print("  --- Propulsion ---")
    print(_fmt_cg("Battery",               cg['battery']))
    print(_fmt_cg("Front motors",          cg['motors']))
    print(_fmt_cg("Rear motor",            cg.get('motor_back', float('nan'))))

    print("  --- Airframe ---")
    print(_fmt_cg("Wing",                  cg.get('wing', float('nan'))))
    print(_fmt_cg("Wing ribs",             cg.get('ribs', float('nan'))))
    print(_fmt_cg("Horizontal tail",       cg.get('tail', float('nan'))))
    print(_fmt_cg("Fuselage",              cg['fuselage']))
    print(_fmt_cg("Tail boom ref.",        cg['pvc_tubes']))

    print("  --- Rods ---")
    print(_fmt_cg("Wing spar rod",         cg['rod_spar']))
    print(_fmt_cg("Wing aileron rod",      cg['rod_aileron']))
    print(_fmt_cg("Tail (boom) rod",       cg['tail_rod']))
    print(_fmt_cg("HT spar rod",           cg.get('ht_spar', float('nan'))))
    print(_fmt_cg("HT elevator rod",       cg.get('ht_rud',  float('nan'))))
    print(_fmt_cg("VT spar rod",           cg.get('vt_spar', float('nan'))))
    print(_fmt_cg("VT rudder rod",         cg.get('vt_rud',  float('nan'))))
    print(_fmt_cg("Connector spar",        cg.get('rod_connector_spar', float('nan'))))
    print(_fmt_cg("Connector aileron",     cg.get('rod_connector_aileron', float('nan'))))
    print(_fmt_cg("Connector payload",     cg.get('rod_connector_payload', float('nan'))))

    print("  --- Other ---")
    print(_fmt_cg("Servos (avg)",          cg.get('servos',         float('nan'))))
    print(_fmt_cg("Sensors",               cg.get('sensors',        float('nan'))))
    print(_fmt_cg("Wiring (wing)",         cg.get('wiring_wing',    float('nan'))))
    print(_fmt_cg("Wiring (tail)",         cg.get('wiring_tail',    float('nan'))))
    print(_fmt_cg("Wiring (signal)",       cg.get('wiring_signal',  float('nan'))))
    print(_fmt_cg("Glass sheet",           cg.get('glass_sheet',    float('nan'))))

    print(f"\n  {'Overall CG':<28}: {cg['overall']:+8.4f}  m from LEMAC"
          f"  ({cg['overall'] / sizing.c_root:6.2%} of root chord)")

    # ------------------------------------------------------------------
    # y-CG breakdown (vertical axis)
    # ------------------------------------------------------------------
    z_cg = result.y_cg

    print("\n========== CENTER OF GRAVITY (y, vertical axis) ==========")
    print("  Sign convention: +y upward from bottom of airfoil/fuselage")
    print("  --- Propulsion ---")
    print(_fmt_cg("Battery",               z_cg['battery']))
    print(_fmt_cg("Front motors",          z_cg['motors']))
    print(_fmt_cg("Rear motor",            z_cg.get('motor_back', float('nan'))))

    print("  --- Airframe ---")
    print(_fmt_cg("Wing",                  z_cg.get('wing', float('nan'))))
    print(_fmt_cg("Wing ribs",             z_cg.get('ribs', float('nan'))))
    print(_fmt_cg("Wing AC (thrust ref)",  z_cg.get('wing_ac', float('nan'))))
    print(_fmt_cg("Horizontal tail",       z_cg.get('hor_tail', float('nan'))))
    print(_fmt_cg("Vertical tail",         z_cg.get('ver_tail', float('nan'))))
    print(_fmt_cg("Fuselage",              z_cg.get('fuselage', float('nan'))))
    print(_fmt_cg("Tail boom",             z_cg.get('pvc_tubes', float('nan'))))
    print(_fmt_cg("Tail boom bottom",      z_cg.get('pvc_tube_bottom', float('nan'))))
    print(_fmt_cg("Battery bottom",        z_cg.get('battery_bottom', float('nan'))))

    print("  --- Rods ---")
    print(_fmt_cg("Wing spar rod",         z_cg.get('rod_spar',    float('nan'))))
    print(_fmt_cg("Wing aileron rod",      z_cg.get('rod_aileron', float('nan'))))
    print(_fmt_cg("Tail (boom) rod",       z_cg.get('tail_rod',    float('nan'))))
    print(_fmt_cg("HT spar rod",           z_cg.get('vt_spar',     float('nan'))))
    print(_fmt_cg("HT elevator rod",       z_cg.get('vt_rud',      float('nan'))))
    print(_fmt_cg("VT spar rod",           z_cg.get('vt_spar',     float('nan'))))
    print(_fmt_cg("VT rudder rod",         z_cg.get('vt_rud',      float('nan'))))
    print(_fmt_cg("Connector spar",        z_cg.get('rod_connector_spar', float('nan'))))
    print(_fmt_cg("Connector aileron",     z_cg.get('rod_connector_aileron', float('nan'))))
    print(_fmt_cg("Connector payload",     z_cg.get('rod_connector_payload', float('nan'))))

    print("  --- Other ---")
    print(_fmt_cg("Servos (avg)",          z_cg.get('servos',        float('nan'))))
    print(_fmt_cg("Servo (front)",         z_cg.get('servo_front',   float('nan'))))
    print(_fmt_cg("Servo (aileron)",       z_cg.get('servo_aileron', float('nan'))))
    print(_fmt_cg("Servo (rear)",          z_cg.get('servo_rear',    float('nan'))))
    print(_fmt_cg("Servo (HT)",            z_cg.get('servo_ht',      float('nan'))))
    print(_fmt_cg("Servo (VT)",            z_cg.get('servo_vt',      float('nan'))))
    print(_fmt_cg("Sensors",               z_cg.get('sensors',       float('nan'))))
    print(_fmt_cg("Wiring (wing)",         z_cg.get('wiring_wing',   float('nan'))))
    print(_fmt_cg("Wiring (tail)",         z_cg.get('wiring_tail',   float('nan'))))
    print(_fmt_cg("Wiring (signal)",       z_cg.get('wiring_signal', float('nan'))))

    print(f"\n  {'Overall CG (y)':<28}: {z_cg['overall']:+8.4f}  m")

    # ------------------------------------------------------------------
    # Inertia
    # ------------------------------------------------------------------
    inertia = calculate_mass_moment_of_inertia(
        sizing=sizing,
        fus=result.fus,
        structure=result.struct,
        masses=masses,
        cg=cg,
        z_cg=z_cg,
        n_props=result.propulsion.inputs.n_props,
    )
    print("\n========== MASS MOMENT OF INERTIA ==========")
    print(f"  Ixx roll  : {inertia.ixx:.4f} kg*m^2")
    print(f"  Iyy pitch : {inertia.iyy:.4f} kg*m^2")
    print(f"  Izz yaw   : {inertia.izz:.4f} kg*m^2")
    print(f"  Ixy       : {inertia.ixy:.4f} kg*m^2")
    print(f"  Ixz       : {inertia.ixz:.4f} kg*m^2")
    print(f"  Iyz       : {inertia.iyz:.4f} kg*m^2")

    print("\n========== DRAG BUILDUP ==========")
    drag_buildup.summary(drag)

    # ----- LLT at the required CL -----
    print(f"\nRequired wing CL for L = W : {result.cl_req:.4f}")
    print(f"\nLLT @ CL_target = {result.cl_req:.4f}:")
    print(f"  alpha_root           : {np.degrees(result.llt.alpha_root):.2f}°")
    print(f"  max section Cl_local : {result.max_cl_local:.3f}  "
          f"(polar Cl_max = {result.cl_max:.3f})")
    if result.lift_achievable:
        print(f"  ✓ Lift achievable — section margin "
              f"{1 - result.max_cl_local / result.cl_max:.1%}")
    else:
        print(f"  ✗ Lift NOT achievable — section Cl exceeds polar by "
              f"{result.max_cl_local - result.cl_max:.3f}")

    failure_margin = result.cl_max_wing - sizing.CL_one_drone_failure
    print(f"\nOne-drone-failure wing CL:")
    print(f"  CL required          : {sizing.CL_one_drone_failure:.4f}")
    print(f"  Wing CL_max          : {result.cl_max_wing:.4f}")
    print(f"  Margin               : {failure_margin:+.4f}")
    print(f"  Status               : {'OK' if result.failure_lift_achievable else 'FAIL'}")

    # ----- α sweep summary -----
    print("\nSweeping α to build drag polar…")
    LD_drone = result.cl_sweep / result.cd_drone_sweep
    LD_full = result.cl_sweep / result.cd_full_sweep
    i_drone = int(np.argmax(LD_drone))
    i_full = int(np.argmax(LD_full))
    print(f"  Drone only      : max L/D = {LD_drone[i_drone]:.2f} "
          f"at CL = {result.cl_sweep[i_drone]:.3f}")
    print(f"  Drone + payload : max L/D = {LD_full[i_full]:.2f} "
          f"at CL = {result.cl_sweep[i_full]:.3f}")
    print(f"  Operating CL    : {result.cl_req:.3f}")


def print_final_drag(result: PipelineResult) -> None:
    print("\n========== FINAL SUMMARY ==========")
    print(f"  Total mass   : {result.masses['total']:.4f}  kg")
    print(f"  Total energy : {result.propulsion.E_total / 3600.0:.2f}  Wh")
    _print_vehicle_block(result)

    tl = result.tail_loading
    total_cd = result.cd_full_buildup
    qS = result.sizing.q_cruise * result.sizing.Sw
    total_drag = total_cd * qS

    def print_drag_line(label: str, cd: float) -> None:
        pct = cd / total_cd if total_cd > 0.0 else 0.0
        print(f"  {label:<34}: CD={cd:.5f}  D={cd * qS:8.3f} N  ({pct:6.2%})")

    print("\n========== CRUISE DRAG OPERATING POINT ==========")
    print(f"  V_cruise                            : {result.sizing.inputs.V_cruise:.2f} m/s")
    print(f"  rho                                 : {result.sizing.rho:.4f} kg/m^3")
    print(f"  q                                   : {result.sizing.q_cruise:.3f} Pa")
    print(f"  S_ref                               : {result.sizing.Sw:.4f} m^2")
    print(f"  q*S_ref                             : {qS:.3f} N")
    print(f"  Operating CL                        : {result.cl_req:.5f}")
    print_drag_line("CD0 wing", result.drag.CD0_wing)
    print_drag_line("CD0 horizontal tail", result.drag.CD0_tail_h)
    print_drag_line("CD0 vertical tail", result.drag.CD0_tail_v)
    print_drag_line("CD0 fuselage", result.drag.CD0_fus)
    print_drag_line("CD0 tail boom", result.drag.CD0_boom)
    print_drag_line("CD0 subtotal", result.drag.CD0)
    print_drag_line("Wing induced", result.llt.CD_i)
    print_drag_line("Tail induced, S_w ref", result.cd_i_tail)
    print_drag_line("Payload", result.cd_payload)
    print(f"  {'TOTAL cruise drag':<34}: CD={total_cd:.5f}  D={total_drag:8.3f} N  (100.00%)")
    print(
        f"  {'Propulsion thrust demand':<34}: "
        f"{result.propulsion.thrust_cruise_per_prop * result.propulsion.inputs.n_props:8.3f} N"
    )

    print(f"Tail trim (with elevator/downwash trim):")
    print(f"  CL_tail required (trim)             : {tl['CL_tail']:+.4f}")
    print(f"  e_tail                              : {tl['e_tail']:.4f}")
    print(f"  AR_tail                             : {tl['AR_tail']:.3f}")
    print(f"  CD_i_tail (S_t ref)                 : {tl['CD_i_tail']:.5f}")
    print(f"Final L/D at operating CL             : "
          f"{result.cl_req / total:.2f}")

    sc = result.scissor
    ShS_stab_now = float(np.interp(sc.x_cg_current, sc.x_cg, sc.ShS_stab))
    ShS_ctrl_now = float(np.interp(sc.x_cg_current, sc.x_cg, sc.ShS_ctrl))
    ShS_req_now = max(ShS_stab_now, ShS_ctrl_now)
    print(f"\nScissor (stability + controllability):")
    print(f"  b_f (fuselage width)                : {sc.b_f:.4f}  m")
    print(f"  S_net (exposed wing area)           : {sc.S_net:.4f}  m²")
    print(f"  r = 2·l_h/b                         : {sc.r:.4f}")
    print(f"  CL_α,w  (wing, LLT)                 : {sc.CL_alpha_w:.4f}  /rad")
    print(f"  CL_α,h  (tail, LLT)                 : {sc.CL_alpha_h:.4f}  /rad")
    print(f"  CL_α,A-h (wing + fuselage)          : {sc.CL_alpha_A_h:.4f}  /rad")
    print(f"  dε/dα   (Slingerland, Λ=0, m_tv=0)  : {sc.dep_da:.4f}")
    print(f"  CL_h    (controllability)           : {sc.CL_h:+.4f}")
    print(f"  CL_A-h  (controllability)           : {sc.CL_A_h:.4f}")
    print(f"  Cm_ac   (3D wing, at cruise α)      : {sc.Cm_ac:+.4f}")
    print(f"  Cm_payload trim                     : {sc.Cm_payload:+.4f}")
    print(f"  SM target                           : {sc.SM:.3f}  ({sc.SM:.1%} MAC)")
    print(f"  V_h/V                               : {sc.Vh_V:.3f}")
    print(f"  Required S_h/S — stability          : {ShS_stab_now:.4f}")
    print(f"  Required S_h/S — controllability    : {ShS_ctrl_now:.4f}")
    print(f"  Required S_h/S — binding            : {ShS_req_now:.4f}")
    scissor_tol = 1.0e-4
    scissor_margin = sc.ShS_current - ShS_req_now
    print(f"  Current  S_h/S                      : {sc.ShS_current:.4f}  "
          f"({'FEASIBLE' if scissor_margin >= -scissor_tol else 'INFEASIBLE'}, "
          f"margin = {scissor_margin:+.4f})")
