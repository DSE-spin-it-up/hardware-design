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


def _fmt_cg(label: str, value: float, width: int = 28) -> str:
    return f"  {label:<{width}}: {value:+8.4f}  m"


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
    print(_fmt_mass("Horizontal tail",      masses['hor_tail']))
    print(_fmt_mass("Vertical tail",        masses['ver_tail']))
    print(_fmt_mass("Fuselage (shell)",     masses['fuselage']))
    print(_fmt_mass("Structural tube",      masses.get('pvc_tubes', 0.0)))

    print("  --- Rods ---")
    print(_fmt_mass("Wing spar rod",        masses['rod_spar']))
    print(_fmt_mass("Wing aileron rod",     masses['rod_aileron']))
    print(_fmt_mass("Tail (boom) rod",      masses['tail_rod']))
    print(_fmt_mass("HT spar rod",          masses.get('ht_spar', 0.0)))
    print(_fmt_mass("HT elevator rod",      masses.get('ht_rud', 0.0)))
    print(_fmt_mass("VT spar rod",          masses.get('vt_spar', 0.0)))
    print(_fmt_mass("VT rudder rod",        masses.get('vt_rud', 0.0)))

    print("  --- Glass sheet ---")
    print(_fmt_mass("Glass sheet (wing)",   masses.get('glass_sheet_wing',   0.0)))
    print(_fmt_mass("Glass sheet (HT)",     masses.get('glass_sheet_tail_h', 0.0)))
    print(_fmt_mass("Glass sheet (VT)",     masses.get('glass_sheet_tail_v', 0.0)))
    print(_fmt_mass("Glass sheet (total)",  masses.get('glass_sheet',        0.0)))

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
    print(_fmt_cg("Horizontal tail",       cg.get('tail', float('nan'))))
    print(_fmt_cg("Fuselage",              cg['fuselage']))
    print(_fmt_cg("Structural tube",       cg['pvc_tubes']))

    print("  --- Rods ---")
    print(_fmt_cg("Wing spar rod",         cg['rod_spar']))
    print(_fmt_cg("Wing aileron rod",      cg['rod_aileron']))
    print(_fmt_cg("Tail (boom) rod",       cg['tail_rod']))
    print(_fmt_cg("HT spar rod",           cg.get('ht_spar', float('nan'))))
    print(_fmt_cg("HT elevator rod",       cg.get('ht_rud',  float('nan'))))
    print(_fmt_cg("VT spar rod",           cg.get('vt_spar', float('nan'))))
    print(_fmt_cg("VT rudder rod",         cg.get('vt_rud',  float('nan'))))

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
    print(_fmt_cg("Wing AC (thrust ref)",  z_cg.get('wing_ac', float('nan'))))
    print(_fmt_cg("Horizontal tail",       z_cg.get('hor_tail', float('nan'))))
    print(_fmt_cg("Vertical tail",         z_cg.get('ver_tail', float('nan'))))
    print(_fmt_cg("Fuselage",              z_cg.get('fuselage', float('nan'))))
    print(_fmt_cg("Structural tube",       z_cg.get('pvc_tubes', float('nan'))))
    print(_fmt_cg("Tube bottom",           z_cg.get('pvc_tube_bottom', float('nan'))))
    print(_fmt_cg("Battery bottom",        z_cg.get('battery_bottom', float('nan'))))

    print("  --- Rods ---")
    print(_fmt_cg("Wing spar rod",         z_cg.get('rod_spar',    float('nan'))))
    print(_fmt_cg("Wing aileron rod",      z_cg.get('rod_aileron', float('nan'))))
    print(_fmt_cg("Tail (boom) rod",       z_cg.get('tail_rod',    float('nan'))))
    print(_fmt_cg("HT spar rod",           z_cg.get('vt_spar',     float('nan'))))
    print(_fmt_cg("HT elevator rod",       z_cg.get('vt_rud',      float('nan'))))
    print(_fmt_cg("VT spar rod",           z_cg.get('vt_spar',     float('nan'))))
    print(_fmt_cg("VT rudder rod",         z_cg.get('vt_rud',      float('nan'))))

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
    tl = result.tail_loading
    total = result.cd_full_buildup
    print(f"\nFull-buildup CD                       : {total:.5f}")
    print(f"  CD0 (buildup)                       : {result.drag.CD0:.5f}  ({result.drag.CD0 / total:6.2%})")
    print(f"  CD_i  (LLT, wing)                   : {result.llt.CD_i:.5f}  ({result.llt.CD_i / total:6.2%})")
    print(f"  CD_i  (LLT, tail, S_w ref)          : {result.cd_i_tail:.5f}  ({result.cd_i_tail / total:6.2%})")
    print(f"  CD_payload                          : {result.cd_payload:.5f}  ({result.cd_payload / total:6.2%})")
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
