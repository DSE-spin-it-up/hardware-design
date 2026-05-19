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
from sizing import aileron, fuselage, wing
from structures import rods


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
    aileron.summary(result.control_surface)
    print("\n========== STRUCTURE ==========")
    rods.summary(result.struct)

    print("\n========== MASS ESTIMATES ==========")
    print(f"  Battery mass : {masses['battery']:.3f} kg")
    print(f"  Motor mass   : {masses['motors']:.3f} kg")
    print(f"  Prop mass    : {masses['props']:.3f} kg")
    print(f"  Wing mass    : {masses['wing']:.3f} kg")
    print(f"  Tail mass    : {masses['tail']:.3f} kg")
    if 'glass_sheet_wing' in masses:
            print(f"  Glass sheet (wing): {masses['glass_sheet_wing']:.3f} kg")
    if 'glass_sheet_tail' in masses:
            print(f"  Glass sheet (tail): {masses['glass_sheet_tail']:.3f} kg")
    if 'glass_sheet' in masses:
            print(f"  Glass sheet (total): {masses['glass_sheet']:.3f} kg")
    print(f"  Spar rod mass   : {masses['rod_spar']:.3f} kg")
    print(f"  Aileron rod mass: {masses['rod_aileron']:.3f} kg")
    print(f"  Tail rod mass: {masses['tail_rod']:.3f} kg")
    print(f"  VT spar mass: {masses['vt_spar']:.3f} kg")
    print(f"  VT rud mass: {masses['vt_rud']:.3f} kg")
    print(f"  Fuselage mass: {masses['fuselage']:.3f} kg")
    if 'pvc_tubes' in masses:
        print(f"  PVC tubes    : {masses['pvc_tubes']:.3f} kg")
    print(f"  Total mass   : {masses['total']:.3f} kg")

    print("\n========== CENTER OF GRAVITY ==========")
    print(f"  Fuselage CG     : {cg['fuselage']:8.4f}  m from LEMAC")
    print(f"  Battery CG      : {cg['battery']:8.4f}  m from LEMAC")
    print(f"  Motors CG       : {cg['motors']:8.4f}  m from LEMAC")
    print(f"  Spar rod CG     : {cg['rod_spar']:8.4f}  m from LEMAC")
    print(f"  Aileron rod CG  : {cg['rod_aileron']:8.4f}  m from LEMAC")
    print(f"  Tail rod CG     : {cg['tail_rod']:8.4f}  m from LEMAC")
    print(f"  VT spar rod CG  : {cg['vt_spar']:8.4f}  m from LEMAC")
    print(f"  VT rud rod CG   : {cg['vt_rud']:8.4f}  m from LEMAC")
    print(f"  Tail CG         : {cg['tail']:8.4f}  m from LEMAC")
    print(f"  PVC tubes CG    : {cg['pvc_tubes']:8.4f}  m from LEMAC")
    print(f"  Overall CG      : {cg['overall']:8.4f}  m from LEMAC")
    print(f"                    ({cg['overall'] / sizing.c_root:6.2%} of wing chord)")

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
    print(f"Tail trim (no downwash):")
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
    print(f"  SM target                           : {sc.SM:.3f}  ({sc.SM:.1%} MAC)")
    print(f"  V_h/V                               : {sc.Vh_V:.3f}")
    print(f"  Required S_h/S — stability          : {ShS_stab_now:.4f}")
    print(f"  Required S_h/S — controllability    : {ShS_ctrl_now:.4f}")
    print(f"  Required S_h/S — binding            : {ShS_req_now:.4f}")
    print(f"  Current  S_h/S                      : {sc.ShS_current:.4f}  "
          f"({'FEASIBLE' if sc.ShS_current >= ShS_req_now else 'INFEASIBLE'}, "
          f"margin = {sc.ShS_current - ShS_req_now:+.4f})")
