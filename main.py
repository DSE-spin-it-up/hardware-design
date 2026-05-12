"""Design pipeline entry point.

Edit the design knobs below and run `python main.py`. The pipeline:
  1. Initial sizing with a guessed wing Cd0
  2. Battery + propulsion sizing
  3. Re-estimate Cd0 from fuselage (PLACEHOLDER — teammate)
  4. Re-run sizing + electrical with the refined Cd0
  5. Pick airfoil → LLT solve at the required wing CL
  6. Wing CL/CD vs CL sweep; check whether operating CL sits near max L/D
  7. Full-buildup CD / final L/D (PLACEHOLDER — teammate)
"""
from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from aerodynamics import drag_estimates
from aerodynamics.airfoil_polar import AirfoilPolar, get_airfoil_polar
from aerodynamics.airfoil_shape import airfoil_thickness_to_chord
from aerodynamics.drag_estimates import DragInputs, DragResult
from aerodynamics.llt_solver import FlightCondition, LLTResult, WingGeometry, solve_llt
from sizing import electrical_system, fuselage, initial_sizing, mass_estimates
from sizing.electrical_system import ElectricalInputs, ElectricalResult
from sizing.fuselage import FuselageInputs, FuselageResult
from sizing.initial_sizing import SizingInputs, SizingResult

# ============================================================
# DESIGN KNOBS
# ============================================================

SIZING = SizingInputs(
    m_payload=60,
    m_drone_empty=10,
    n_drones=3,
    b=3.0,
    AR=7.5,
    lam=1.0,
    Cd0=0.045,            # initial guess; refined after fuselage sizing
    S_payload=0.25,
    Cd_payload=1.0,
    h_cruise=100,
    V_cruise=20,
    R=20000,
    foam_density=48,
    Vv=0.04,
    Vh=0.50,
    ARt=4.5,
    lam_t=1.0,
)

ELECTRICAL = ElectricalInputs(
    n_props=2,
    eff_motor=0.8,
    eff_prop=0.7,
    T_W_to=2.0,
    J=0.4,
    C_t=0.04,
    D_prop=20 * 0.0254,
    n_cells=6,
    voltage_cell=3.7,
    battery_density=250,
)

# Wing airfoil: .dat path or NACA digits (e.g. "2412"). If None, the pipeline prompts at runtime.
AIRFOIL: str | None = "airfoils/MH112.dat"
# AIRFOIL: str | None = "4412"

# Tail airfoil: .dat path or NACA digits. Symmetric sections are typical.
TAIL_AIRFOIL: str = "airfoils/NACA0010.dat"

# Alpha sweep used to build the wing drag polar for the CL/CD plot [deg].
ALPHA_SWEEP_DEG = (-2.0, 12.0, 30)


# ============================================================
# PLACEHOLDERS — replace as teammates' modules land
# ============================================================

def estimate_cd0(
    sizing: SizingResult,
    fus: FuselageResult,
    wing_airfoil: str,
    tail_airfoil: str = "airfoils/NACA0010.dat",
) -> DragResult:
    """Component drag buildup for wing + tail + fuselage."""
    return drag_estimates.run(
        sizing,
        fus,
        DragInputs(wing_airfoil=wing_airfoil, tail_airfoil=tail_airfoil),
    )


def full_drag_estimate(
    sizing: SizingResult,
    drag: DragResult,
    llt: LLTResult,
) -> float:
    """Total wing-level CD at the operating point: CD0 buildup + induced drag."""
    return drag.CD0 + llt.CD_i


# ============================================================
# PIPELINE STEPS
# ============================================================

def resolve_airfoil(airfoil: str | None) -> str:
    if airfoil is not None:
        return airfoil
    return input("Enter airfoil (.dat path or NACA digits): ").strip()


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


def plot_wing_ld(
    CL: np.ndarray,
    CD: np.ndarray,
    CL_op: float,
    airfoil_name: str,
    out_path: str = "wing_LD.png",
) -> None:
    LD = CL / CD
    idx_max = int(np.argmax(LD))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(CL, LD, "o-", label="wing L/D")
    ax.axvline(CL_op, color="r", ls="--", label=f"operating CL = {CL_op:.3f}")
    ax.plot(
        CL[idx_max], LD[idx_max], "g*", ms=14,
        label=f"max L/D = {LD[idx_max]:.1f} @ CL = {CL[idx_max]:.2f}",
    )
    ax.set_xlabel("Wing CL")
    ax.set_ylabel("Wing L/D")
    ax.set_title(f"Wing drag polar — airfoil {airfoil_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=150)
    # print(f"  L/D plot saved to {out_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    electrical_inputs = ELECTRICAL

    # ----- Step 0: resolve airfoil → read t/c from its geometry -----
    airfoil = resolve_airfoil(AIRFOIL)
    tc = airfoil_thickness_to_chord(airfoil)
    print(f"Airfoil: {airfoil}  (t/c = {tc:.4f})")
    sizing_inputs = SIZING

    # ----- Step 1: initial sizing with guessed Cd0 -----
    sizing = initial_sizing.run(sizing_inputs, t_over_c_root=tc)

    # ----- Step 2: battery + propulsion -----
    electrical = electrical_system.run(sizing, electrical_inputs)

    # ----- Step 3: size fuselage and refine CD0 from a component buildup -----
    fus = fuselage.run(sizing)
    drag = estimate_cd0(sizing, fus, wing_airfoil=airfoil, tail_airfoil=TAIL_AIRFOIL)
    print(f">>> CD0 estimate (pass 1, Cd0_guess={sizing_inputs.Cd0:.5f}) : {drag.CD0:.5f}")
    sizing_inputs = dataclasses.replace(sizing_inputs, Cd0=drag.CD0)

    # ----- Step 4: re-run sizing + electrical with refined Cd0 -----
    # (One re-pass for now; turn into a `while not converged` loop once the
    # CD0 estimator and sizing converge non-trivially.)
    sizing = initial_sizing.run(sizing_inputs, t_over_c_root=tc)
    electrical = electrical_system.run(sizing, electrical_inputs)
    fus = fuselage.run(sizing)
    drag = estimate_cd0(sizing, fus, wing_airfoil=airfoil, tail_airfoil=TAIL_AIRFOIL)
    print(f">>> CD0 estimate (pass 2, refined)                          : {drag.CD0:.5f}")

    print("========== INITIAL SIZING ==========")
    initial_sizing.summary(sizing)
    print("\n========== ELECTRICAL SYSTEM ==========")
    electrical_system.summary(electrical)
    print("\n========== FUSELAGE ==========")
    fuselage.summary(fus)
    print("\n========== MASS ESTIMATES ==========")
    masses = mass_estimates.total_mass(
        sizing=sizing,
        electrical=electrical,
        airfoil_path=airfoil,
        tail_airfoil_path=TAIL_AIRFOIL,
    )
    print(f"  Battery mass : {masses['battery']:.3f} kg")
    print(f"  Motor mass   : {masses['motors']:.3f} kg")
    print(f"  Wing mass    : {masses['wing']:.3f} kg")
    print(f"  Tail mass    : {masses['tail']:.3f} kg")
    print(f"  Fuselage mass: {masses['fuselage']:.3f} kg")
    print(f"  Total mass   : {masses['total']:.3f} kg")
    print("\n========== DRAG BUILDUP ==========")
    drag_estimates.summary(drag)

    # ----- Step 5: LLT at the required CL -----
    CL_req = sizing.CL
    print(f"\nRequired wing CL for L = W : {CL_req:.4f}")

    Re_ref = sizing.rho * sizing.inputs.V_cruise * sizing.c / 1.7894e-5
    polar = get_airfoil_polar(
        airfoil, Re=Re_ref, M=0.0, alpha_range=(-5.0, 15.0, 0.5), use_cache=True
    )
    Cl_max = float(np.max(polar.Cl))
    print(f"Polar: Cl_alpha = {polar.Cl_alpha:.3f}/rad, "
          f"alpha_L0 = {np.degrees(polar.alpha_L0):.2f}°, Cl_max = {Cl_max:.3f}")

    llt = llt_at_cl(sizing, polar, CL_target=CL_req)
    max_Cl_local = float(np.max(np.abs(llt.Cl_local)))
    print(f"\nLLT @ CL_target = {CL_req:.4f}:")
    print(f"  alpha_root           : {np.degrees(llt.alpha_root):.2f}°")
    print(f"  max section Cl_local : {max_Cl_local:.3f}  (polar Cl_max = {Cl_max:.3f})")
    if max_Cl_local <= Cl_max:
        print(f"  ✓ Lift achievable — section margin {1 - max_Cl_local / Cl_max:.1%}")
    else:
        print(f"  ✗ Lift NOT achievable — section Cl exceeds polar by "
              f"{max_Cl_local - Cl_max:.3f}")

    # ----- Step 6: wing CL/CD sweep → plot operating point vs max L/D -----
    print("\nSweeping α to build wing drag polar…")
    CL_sweep, CD_sweep = wing_drag_polar(sizing, polar, ALPHA_SWEEP_DEG)
    LD_sweep = CL_sweep / CD_sweep
    idx_max = int(np.argmax(LD_sweep))
    print(f"  Max L/D = {LD_sweep[idx_max]:.2f} at CL = {CL_sweep[idx_max]:.3f}  "
          f"(operating CL = {CL_req:.3f})")
    plot_wing_ld(CL_sweep, CD_sweep, CL_op=CL_req, airfoil_name=polar.name)

    # ----- Step 7: full drag estimate = CD0 buildup + induced drag -----
    CD_full = full_drag_estimate(sizing, drag, llt)
    print(f"\nFull-buildup CD                       : {CD_full:.5f}")
    print(f"  CD0 (buildup)                       : {drag.CD0:.5f}")
    print(f"  CD_i  (LLT)                         : {llt.CD_i:.5f}")
    print(f"Final L/D at operating CL             : {CL_req / CD_full:.2f}")


if __name__ == "__main__":
    main()
