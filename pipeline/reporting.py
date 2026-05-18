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
    print(f"  Spar rod mass   : {masses['rod_spar']:.3f} kg")
    print(f"  Aileron rod mass: {masses['rod_aileron']:.3f} kg")
    print(f"  Tail rod mass: {masses['tail_rod']:.3f} kg")
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
    print(f"\nFull-buildup CD                       : {result.cd_full_buildup:.5f}")
    print(f"  CD0 (buildup)                       : {result.drag.CD0:.5f}")
    print(f"  CD_i  (LLT)                         : {result.llt.CD_i:.5f}")
    print(f"  CD_payload                          : {result.cd_payload:.5f}")
    print(f"Final L/D at operating CL             : "
          f"{result.cl_req / result.cd_full_buildup:.2f}")
