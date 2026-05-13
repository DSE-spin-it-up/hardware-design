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
from sizing import control_surface_sizing, propulsion_sizing, fuselage, initial_sizing, mass_estimates
from sizing.control_surface_sizing import ControlSurfaceInputs
from sizing.propulsion_sizing import PropulsionInputs
from sizing.fuselage import FuselageInputs, FuselageResult
from sizing.initial_sizing import SizingInputs, SizingResult

# ============================================================
# DESIGN KNOBS
# ============================================================

SIZING = SizingInputs(
    m_payload=60,
    m_drone_empty=10,
    n_drones=4,
    b=3.0,
    AR=7.5,
    lam=1.0,
    Cd0=0.03,            # initial guess; refined after fuselage sizing
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

PROPULSION = PropulsionInputs(
    csv_prop="data/20x10E_performance.csv",
    n_props=2,
    D_prop=20 * 0.0254,  # [m] — derived from csv_prop name if None
    prop_mass=0.1,       # [kg] — mass per propeller
    # Motor / drivetrain
    eff_motor=0.8,
    T_W_to=2.0,          # thrust-to-weight for take-off / VTOL [-]
    throttle_max=0.9,    # design throttle cap at the sizing RPM [-]
    # Climb segment
    climb_rate=10 / 3,   # [m/s]
    t_climb=30,          # [s]
    # Battery
    n_cells=6,
    voltage_cell=3.7,    # [V]
    battery_density=250, # [Wh/L]
    # Propeller solver tuning (shared between cruise & climb)
    cruise_rpm_init=10000,
    climb_rpm_init=10000,
    rpm_tol=1.0,
    thrust_tol=0.5,
    max_iter=200,
    rpm_step=50.0,
    # VTOL/hover solver tuning — looser thrust tol because the table-step
    # resolution (rpm_step) easily exceeds 0.5 N near hover RPM.
    vtol_rpm_init=5000,
    vtol_thrust_tol=2.0,
    vtol_max_iter=400,
    vtol_rpm_step=25.0,
    # Verbosity — solver prints per-iteration tables; off by default so the
    # main.py iteration loop stays readable.
    verbose=True,
)

# Battery dimensions: off-the-shelf envelope unless both aspect ratios are > 0,
# in which case dimensions are recomputed from the required battery volume.
# AR_lw / AR_lh below match the off-the-shelf 0.212 × 0.090 × 0.060 cell so the
# starting shape is preserved but the cell scales with the propulsion-sized volume.
FUSELAGE = FuselageInputs(
    battery_length=0.212,
    battery_width=0.090,
    battery_height=0.060,
    AR_lw=0.212 / 0.090,  # length / width  (≈ 2.356)
    AR_lh=0.212 / 0.060,  # length / height (≈ 3.533)
    housing_factor=1.5,
    casing_factor=1.1,
)

CONTROL_SURFACE = ControlSurfaceInputs(
    cl_alpha=2 * np.pi,
    cd0_section=0.04,
    roll_req_deg=60.0,
    max_da_deg=10.0,
    max_y_frac=0.8,
    c_aileron_to_c_wing=0.3,
)

# Wing airfoil: .dat path or NACA digits (e.g. "2412"). If None, the pipeline prompts at runtime.
AIRFOIL: str | None = "airfoils/MH112.dat"
AIRFOIL: str | None = "airfoils/NACA4412.dat"

# Tail airfoil: .dat path or NACA digits. Symmetric sections are typical.
TAIL_AIRFOIL: str = "airfoils/NACA0010.dat"

# Alpha sweep used to build the wing drag polar for the CL/CD plot [deg].
ALPHA_SWEEP_DEG = (-2.0, 12.0, 30)
# Coarser sweep used inside the iteration loop to find max-L/D CL each pass.
ALPHA_SWEEP_LOOP_DEG = (-2.0, 12.0, 15)

# Sizing↔electrical↔fuselage↔drag↔mass↔wing-area iteration.
#   MASS_CLOSURE = False  → hold m_drone_empty at SIZING.m_drone_empty (requirement).
#   MASS_CLOSURE = True   → feed total drone mass back into m_drone_empty each pass.
#   SW_CLOSURE   = False  → hold b, AR (and therefore Sw) at the SIZING values.
#   SW_CLOSURE   = True   → resize Sw each pass so the operating CL sits at the max
#                           L/D point of the drone+payload polar; AR is held fixed,
#                           so b = sqrt(AR · Sw) floats.
# Convergence requires every active criterion (CD0 always; m_drone if MASS_CLOSURE;
# Sw if SW_CLOSURE) to be inside its tolerance.
MASS_CLOSURE = True
SW_CLOSURE = True
N_ITER_MAX = 20
CD0_TOL = 1e-4
MASS_TOL = 1e-3   # kg
SW_TOL = 1e-3     # m²


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
    """Total CD at the operating point: CD0 buildup + induced + payload bluff-body."""
    s = sizing.inputs
    CD_payload = s.Cd_payload * s.S_payload / (s.n_drones * sizing.Sw)
    return drag.CD0 + llt.CD_i + CD_payload


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


def plot_convergence(
    CD0_history: list[float],
    mass_history: list[float],
    Sw_history: list[float],
    Cd0_guess: float,
    m_drone_guess: float,
    Sw_guess: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    cd_iters = np.arange(len(CD0_history) + 1)
    cd_values = np.array([Cd0_guess, *CD0_history])
    axes[0].plot(cd_iters, cd_values, "o-")
    axes[0].axhline(cd_values[-1], color="g", ls="--",
                    label=f"converged CD0 = {cd_values[-1]:.5f}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("CD0")
    axes[0].set_title("CD0 progression")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    m_iters = np.arange(len(mass_history) + 1)
    m_values = np.array([m_drone_guess, *mass_history])
    axes[1].plot(m_iters, m_values, "o-")
    axes[1].axhline(m_values[-1], color="g", ls="--",
                    label=f"converged m_drone = {m_values[-1]:.3f} kg")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("drone mass [kg]")
    axes[1].set_title("Drone mass progression")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    sw_iters = np.arange(len(Sw_history) + 1)
    sw_values = np.array([Sw_guess, *Sw_history])
    axes[2].plot(sw_iters, sw_values, "o-")
    axes[2].axhline(sw_values[-1], color="g", ls="--",
                    label=f"converged Sw = {sw_values[-1]:.4f} m²")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("wing area Sw [m²]")
    axes[2].set_title("Wing area progression")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    plt.show()


def plot_drone_ld(
    CL: np.ndarray,
    CD_drone: np.ndarray,
    CD_full: np.ndarray,
    CL_op: float,
    airfoil_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, CD, title in [
        (axes[0], CD_drone, "Drone only"),
        (axes[1], CD_full, "Drone + payload"),
    ]:
        LD = CL / CD
        idx_max = int(np.argmax(LD))
        ax.plot(CL, LD, "o-", label="L/D")
        ax.axvline(CL_op, color="r", ls="--", label=f"operating CL = {CL_op:.3f}")
        ax.plot(
            CL[idx_max], LD[idx_max], "g*", ms=14,
            label=f"max L/D = {LD[idx_max]:.1f} @ CL = {CL[idx_max]:.2f}",
        )
        ax.set_xlabel("CL")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("L/D")
    fig.suptitle(f"Drag polar — airfoil {airfoil_name}")
    fig.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    propulsion_inputs = PROPULSION

    # ----- Step 0: resolve airfoil → read t/c from its geometry -----
    airfoil = resolve_airfoil(AIRFOIL)
    tc = airfoil_thickness_to_chord(airfoil)
    print(f"Airfoil: {airfoil}  (t/c = {tc:.4f})")
    sizing_inputs = SIZING

    # ----- Step 1: initial sizing with guessed Cd0 -----
    sizing = initial_sizing.run(sizing_inputs, t_over_c_root=tc)

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

    # ----- Steps 2-4: iterate electrical → fuselage → drag → mass → wing-area → sizing -----
    # Each pass feeds the refined CD0 back into sizing. If MASS_CLOSURE is True,
    # the buildup drone mass is also fed back into m_drone_empty. If SW_CLOSURE
    # is True, the wing area is resized so the operating CL sits at the max-L/D
    # point of the drone+payload polar (AR fixed, b floats).
    mode_bits = []
    mode_bits.append("mass-closure" if MASS_CLOSURE else "fixed m_drone")
    mode_bits.append("Sw-closure" if SW_CLOSURE else "fixed Sw")
    print(f">>> Iterating electrical → fuselage → drag → mass → wing-area → sizing  "
          f"[{', '.join(mode_bits)}]")
    exit_bits = [f"|ΔCD0| < {CD0_TOL:.0e}"]
    if MASS_CLOSURE:
        exit_bits.append(f"|Δm_drone| < {MASS_TOL:.0e} kg")
    if SW_CLOSURE:
        exit_bits.append(f"|ΔSw| < {SW_TOL:.0e} m²")
    print(f"    exit: {' AND '.join(exit_bits)}  (max {N_ITER_MAX} passes)")
    Cd0_guess = sizing_inputs.Cd0
    m_drone_guess = sizing_inputs.m_drone_empty
    Sw_guess = sizing.Sw
    print(f"    iter  0: Cd0={Cd0_guess:.6f}  m_drone={m_drone_guess:.3f}  "
          f"Sw={Sw_guess:.4f}  b={sizing_inputs.b:.3f}")
    CD0_history: list[float] = []
    mass_history: list[float] = []
    Sw_history: list[float] = []
    CD0_prev = Cd0_guess
    m_prev = m_drone_guess
    Sw_prev = Sw_guess
    converged = False
    for it in range(N_ITER_MAX):
        propulsion = propulsion_sizing.run(sizing, propulsion_inputs)
        fus = fuselage.run(
            sizing,
            FUSELAGE,
            battery_volume=propulsion.battery_volume,
            airfoil_path=airfoil,
        )
        if np.isclose(fus.height, FUSELAGE.casing_factor * fus.battery_height):
            print(
                "    WARNING: fuselage height is just casing_factor × battery_height; "
                "airfoil height is not being used for fuselage sizing."
            )
        drag = estimate_cd0(sizing, fus, wing_airfoil=airfoil, tail_airfoil=TAIL_AIRFOIL)
        masses = mass_estimates.total_mass(
            sizing=sizing,
            propulsion=propulsion,
            airfoil_path=airfoil,
            tail_airfoil_path=TAIL_AIRFOIL,
            fuselage_inputs=FUSELAGE,
        )
        m_drone = masses["total"]

        # --- Wing-area closure: resize Sw to put op-point at max-L/D for drone+payload ---
        if SW_CLOSURE:
            CL_sw, CD_wing_sw = wing_drag_polar(sizing, polar, ALPHA_SWEEP_LOOP_DEG)
            CD_nonwing = drag.CD0_tail + drag.CD0_fus
            CD_payload_sw = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
                sizing.inputs.n_drones * sizing.Sw
            )
            LD_full_sw = CL_sw / (CD_wing_sw + CD_nonwing + CD_payload_sw)
            CL_target = float(CL_sw[int(np.argmax(LD_full_sw))])
            m_eff = m_drone if MASS_CLOSURE else sizing_inputs.m_drone_empty
            W = (m_eff + sizing_inputs.m_payload / sizing_inputs.n_drones) * 9.80665
            Sw_new = W / (sizing.q_cruise * CL_target)
            b_new = float(np.sqrt(sizing_inputs.AR * Sw_new))
        else:
            CL_target = None
            b_new = sizing_inputs.b

        # --- Build updated sizing inputs ---
        replace_kwargs: dict = {"Cd0": drag.CD0, "b": b_new}
        if MASS_CLOSURE:
            replace_kwargs["m_drone_empty"] = m_drone
        sizing_inputs = dataclasses.replace(sizing_inputs, **replace_kwargs)
        sizing = initial_sizing.run(sizing_inputs, t_over_c_root=tc)
        control_surface = control_surface_sizing.run(sizing, CONTROL_SURFACE, polar=polar)

        CD0_history.append(drag.CD0)
        mass_history.append(m_drone)
        Sw_history.append(sizing.Sw)
        dCD0 = abs(drag.CD0 - CD0_prev)
        dM = abs(m_drone - m_prev)
        dSw = abs(sizing.Sw - Sw_prev)
        cl_str = f"CL*={CL_target:.3f}  " if CL_target is not None else ""
        print(f"    iter {it + 1:2d}: "
              f"CD0={drag.CD0:.6f}(Δ={dCD0:.1e})  "
              f"m_drone={m_drone:.3f}(Δ={dM:.1e})  "
              f"Sw={sizing.Sw:.4f}(Δ={dSw:.1e})  "
              f"b={sizing.inputs.b:.3f}  {cl_str}"
              f"m_batt={propulsion.battery_mass:.3f}  "
              f"fus={fus.length:.3f}×{fus.width:.3f}×{fus.height:.3f}")
        mass_ok = (dM < MASS_TOL) if MASS_CLOSURE else True
        sw_ok = (dSw < SW_TOL) if SW_CLOSURE else True
        if dCD0 < CD0_TOL and mass_ok and sw_ok:
            converged = True
            break
        CD0_prev = drag.CD0
        m_prev = m_drone
        Sw_prev = sizing.Sw
    if converged:
        ok_bits = [f"|ΔCD0| < {CD0_TOL:.0e}"]
        if MASS_CLOSURE:
            ok_bits.append(f"|Δm| < {MASS_TOL:.0e} kg")
        if SW_CLOSURE:
            ok_bits.append(f"|ΔSw| < {SW_TOL:.0e} m²")
        print(f"    ✓ Converged in {it + 1} iterations ({', '.join(ok_bits)}).")
        if not MASS_CLOSURE:
            print(f"      Buildup m_drone = {m_drone:.3f} kg vs requirement "
                  f"{sizing_inputs.m_drone_empty:.3f} kg.")
    else:
        last_bits = [f"|ΔCD0|={dCD0:.2e}"]
        if MASS_CLOSURE:
            last_bits.append(f"|Δm|={dM:.2e}")
        if SW_CLOSURE:
            last_bits.append(f"|ΔSw|={dSw:.2e}")
        print(f"    ✗ Did NOT converge after {N_ITER_MAX} iterations "
              f"(last {', '.join(last_bits)}).")
    # Final pass with the converged Cd0 so electrical/fus/drag match the latest sizing.
    propulsion = propulsion_sizing.run(sizing, PROPULSION)
    fus = fuselage.run(
        sizing,
        FUSELAGE,
        battery_volume=propulsion.battery_volume,
        airfoil_path=airfoil,
    )
    drag = estimate_cd0(sizing, fus, wing_airfoil=airfoil, tail_airfoil=TAIL_AIRFOIL)
    control_surface = control_surface_sizing.run(sizing, CONTROL_SURFACE, polar=polar)
    plot_convergence(
        CD0_history, mass_history, Sw_history,
        Cd0_guess, m_drone_guess, Sw_guess,
    )

    print("========== INITIAL SIZING ==========")
    initial_sizing.summary(sizing)
    print("\n========== PROPULSION SYSTEM ==========")
    propulsion_sizing.summary(propulsion)
    print("\n========== FUSELAGE ==========")
    fuselage.summary(fus)
    print("\n========== CONTROL SURFACES ==========")
    control_surface_sizing.summary(control_surface)
    print("\n========== MASS ESTIMATES ==========")
    masses = mass_estimates.total_mass(
        sizing=sizing,
        propulsion=propulsion,
        airfoil_path=airfoil,
        tail_airfoil_path=TAIL_AIRFOIL,
    )
    print(f"  Battery mass : {masses['battery']:.3f} kg")
    print(f"  Motor mass   : {masses['motors']:.3f} kg")
    print(f"  Prop mass    : {masses['props']:.3f} kg")
    print(f"  Wing mass    : {masses['wing']:.3f} kg")
    print(f"  Tail mass    : {masses['tail']:.3f} kg")
    print(f"  Rod mass     : {masses['rod']:.3f} kg")
    print(f"  Tail rod mass: {masses['tail_rod']:.3f} kg")
    print(f"  Fuselage mass: {masses['fuselage']:.3f} kg")
    print(f"  Total mass   : {masses['total']:.3f} kg")
    print("\n========== DRAG BUILDUP ==========")
    drag_estimates.summary(drag)

    # ----- Step 5: LLT at the required CL -----
    CL_req = sizing.CL
    print(f"\nRequired wing CL for L = W : {CL_req:.4f}")
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

    # ----- Step 6: CL/CD sweep → plot drone-only and drone+payload L/D -----
    print("\nSweeping α to build drag polar…")
    CL_sweep, CD_wing_sweep = wing_drag_polar(sizing, polar, ALPHA_SWEEP_DEG)
    CD_nonwing = drag.CD0_tail + drag.CD0_fus
    CD_payload = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
        sizing.inputs.n_drones * sizing.Sw
    )
    CD_drone_sweep = CD_wing_sweep + CD_nonwing
    CD_full_sweep = CD_drone_sweep + CD_payload
    LD_drone = CL_sweep / CD_drone_sweep
    LD_full = CL_sweep / CD_full_sweep
    i_drone = int(np.argmax(LD_drone))
    i_full = int(np.argmax(LD_full))
    print(f"  Drone only      : max L/D = {LD_drone[i_drone]:.2f} at CL = {CL_sweep[i_drone]:.3f}")
    print(f"  Drone + payload : max L/D = {LD_full[i_full]:.2f} at CL = {CL_sweep[i_full]:.3f}")
    print(f"  Operating CL    : {CL_req:.3f}")
    plot_drone_ld(
        CL_sweep, CD_drone_sweep, CD_full_sweep,
        CL_op=CL_req, airfoil_name=polar.name,
    )

    # ----- Step 7: full drag estimate = CD0 buildup + induced + payload -----
    CD_full = full_drag_estimate(sizing, drag, llt)
    CD_payload = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
        sizing.inputs.n_drones * sizing.Sw
    )
    print(f"\nFull-buildup CD                       : {CD_full:.5f}")
    print(f"  CD0 (buildup)                       : {drag.CD0:.5f}")
    print(f"  CD_i  (LLT)                         : {llt.CD_i:.5f}")
    print(f"  CD_payload                          : {CD_payload:.5f}")
    print(f"Final L/D at operating CL             : {CL_req / CD_full:.2f}")


if __name__ == "__main__":
    main()
