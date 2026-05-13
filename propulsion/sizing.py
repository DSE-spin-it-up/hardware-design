"""Propulsion sizing using a propeller-table operating-point solver.

Fits the pipeline iteration in `main.py`:

    wing-sizing → prop-sizing → battery_volume → fuselage → drag → wing-sizing
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

from sizing import wing
from propulsion import propeller_solver as prop_solver
from sizing.wing import SizingResult, g0


def _diameter_from_csv_name(csv_path: str) -> float:
    """Parse the leading ``<inches>x...`` from a propeller CSV filename."""
    stem = csv_path.split("/")[-1].split("\\")[-1]
    head = stem.split("x")[0]
    return int(head) * 0.0254

def motor_mass_from_kv(KV: float) -> float:
    """Linear motor-mass fit from KV rating; returns kg."""
    if KV < 100 or KV > 2000:
        raise ValueError(f"KV={KV} out of fit range [100, 2000]")
    if KV < 500:
        slope = (900 - 300) / (500 - 100)
        mass_g = 900 - (KV - 100) * slope
    else:
        slope = (300 - 50) / (2000 - 500)
        mass_g = 300 - (KV - 500) * slope
    return mass_g / 1000


def battery_mass_from_energy(E_cruise: float, n_cells: int, voltage_cell: float) -> float:
    """Battery mass [kg] from required cruise energy [J]."""
    voltage = n_cells * voltage_cell
    C = E_cruise * 1000 / (3600 * voltage)
    if n_cells == 6:
        a, b = 0.3988, 0.8810
    else:
        raise NotImplementedError(f"battery_mass fit not available for n_cells={n_cells}")
    mass_g = a * (C ** b)
    return mass_g / 1000


def battery_volume_from_energy(E_cruise: float, battery_density: float) -> float:
    """Battery volume [m^3] from required cruise energy [J].

    battery_density is in [Wh/L], so volume_L = E_Wh / battery_density.
    """
    volume_L = (E_cruise / 3600) / battery_density
    return volume_L / 1000


@dataclass
class PropulsionInputs:
    # Propeller
    csv_prop: str = "data/20x10E_performance.csv"
    n_props: int = 2
    D_prop: float | None = None  # [m] — derived from csv_prop name if None
    prop_mass: float = 0.1       # [kg] — mass per propeller
    # Motor / drivetrain
    eff_motor: float = 0.8
    T_W_to: float = 2.0          # thrust-to-weight for take-off / VTOL [-]
    throttle_max: float = 0.9    # design throttle cap at the sizing RPM [-]
    # Climb segment
    climb_rate: float = 10 / 3   # [m/s]
    t_climb: float = 30          # [s]
    # Battery
    n_cells: int = 6
    voltage_cell: float = 3.7    # [V]
    battery_density: float = 250 # [Wh/L]
    # Propeller solver tuning (shared between cruise & climb)
    cruise_rpm_init: float = 6000
    climb_rpm_init: float = 9000
    rpm_tol: float = 1.0
    thrust_tol: float = 0.5
    max_iter: int = 200
    rpm_step: float = 50.0
    # VTOL/hover solver tuning — looser thrust tol because the table-step
    # resolution (rpm_step) easily exceeds 0.5 N near hover RPM.
    vtol_rpm_init: float = 5000
    vtol_thrust_tol: float = 2.0
    vtol_max_iter: int = 400
    vtol_rpm_step: float = 25.0
    # Verbosity — solver prints per-iteration tables; off by default so the
    # main.py iteration loop stays readable.
    verbose: bool = False


@dataclass
class PropulsionResult:
    inputs: PropulsionInputs
    voltage_battery: float
    D_prop: float
    # --- Cruise ---
    thrust_cruise_per_prop: float    # [N]
    n_cruise: float                  # [rev/s]
    J_cruise: float
    Cp_cruise: float
    eff_cruise: float
    P_shaft_cruise: float            # total over all props [W]
    P_elec_cruise: float             # electrical input power [W]
    throttle_cruise: float           # [-]
    # --- Climb ---
    CL_climb: float
    V_climb: float                   # [m/s]
    Cd_climb: float
    D_climb: float                   # drag at climb speed [N]
    thrust_climb_per_prop: float     # [N]
    n_climb: float                   # [rev/s]
    J_climb: float
    Cp_climb: float
    eff_climb: float
    P_shaft_climb: float             # total over all props [W]
    P_elec_climb: float              # [W]
    throttle_climb: float
    # --- VTOL / hover ---
    thrust_vtol_per_prop: float
    n_vtol: float
    Cp_vtol: float
    Ct_vtol: float
    P_shaft_vtol: float              # total over all props [W]
    P_elec_vtol: float               # [W]
    throttle_vtol: float
    # --- Motor ---
    KV: float                        # [RPM/V]
    motor_mass: float                # per motor [kg]
    total_motor_mass: float          # [kg]
    # --- Energy & Battery ---
    E_cruise: float                  # [J]
    E_climb: float                   # [J]
    E_total: float                   # [J]
    battery_mass: float              # [kg]
    battery_volume: float            # [m^3]


def _solve_quiet(verbose: bool, fn, *args, **kwargs):
    """Call a propeller_solver function, suppressing its prints when verbose=False."""
    if verbose:
        return fn(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _shaft_power_per_prop(Cp: float, rho: float, n_rps: float, D: float) -> float:
    """P_shaft = Cp · ρ · n³ · D⁵ per propeller."""
    return Cp * rho * n_rps ** 3 * D ** 5


def run(
    sizing: SizingResult,
    inputs: PropulsionInputs | None = None,
) -> PropulsionResult:
    """Size propulsion for cruise, climb, and VTOL given the current sizing.

    Climb speed is taken at the best rate-of-climb CL,
    ``C_L_climb = sqrt(3·Cd0/k)`` (max ``L^{3/2}/D`` for prop aircraft).
    Battery energy covers cruise + climb; VTOL is treated as a sizing
    point for thrust/RPM but its energy contribution is left to the
    caller to add if a hover phase is part of the mission.
    """
    if inputs is None:
        inputs = PropulsionInputs()
    i = inputs
    s = sizing
    si = s.inputs

    D_prop = i.D_prop if i.D_prop is not None else _diameter_from_csv_name(i.csv_prop)
    voltage_battery = i.n_cells * i.voltage_cell

    # ----- Cruise -----
    thrust_cruise_per_prop = s.D / i.n_props
    op_cruise = _solve_quiet(
        i.verbose,
        prop_solver.solve,
        i.csv_prop, thrust_cruise_per_prop, si.V_cruise, D_prop,
        i.cruise_rpm_init, i.rpm_tol, i.thrust_tol, i.max_iter, i.rpm_step,
    )
    if op_cruise is None:
        raise RuntimeError("Cruise prop operating point did not converge.")
    n_cruise = op_cruise["RPM"] / 60.0
    J_cruise = op_cruise["J"]
    Cp_cruise = op_cruise["Cp"]
    eff_cruise = op_cruise["Efficiency"]
    P_shaft_cruise = i.n_props * _shaft_power_per_prop(Cp_cruise, s.rho, n_cruise, D_prop)
    P_elec_cruise = P_shaft_cruise / i.eff_motor

    # ----- Climb (best rate-of-climb CL) -----
    CL_climb = np.sqrt(3 * si.Cd0 / s.k)
    V_climb = np.sqrt(2 * s.m_drone_loaded * g0 / (s.rho * s.Sw * CL_climb))
    Cd_climb = (
        si.Cd0
        + s.k * CL_climb ** 2
        + (si.Cd_payload * si.S_payload / (si.n_drones * s.Sw))
    )
    D_climb = 0.5 * s.rho * V_climb ** 2 * s.Sw * Cd_climb
    thrust_climb_per_prop = (
        D_climb + i.climb_rate * s.m_drone_loaded * g0 / V_climb
    ) / i.n_props
    op_climb = _solve_quiet(
        i.verbose,
        prop_solver.solve,
        i.csv_prop, thrust_climb_per_prop, V_climb, D_prop,
        i.climb_rpm_init, i.rpm_tol, i.thrust_tol, i.max_iter, i.rpm_step,
    )
    if op_climb is None:
        raise RuntimeError("Climb prop operating point did not converge.")
    n_climb = op_climb["RPM"] / 60.0
    J_climb = op_climb["J"]
    Cp_climb = op_climb["Cp"]
    eff_climb = op_climb["Efficiency"]
    P_shaft_climb = i.n_props * _shaft_power_per_prop(Cp_climb, s.rho, n_climb, D_prop)
    P_elec_climb = P_shaft_climb / i.eff_motor

    # ----- VTOL / hover -----
    # Faithful port: VTOL thrust sized to T_W_to · empty weight (matches
    # propulsion-design). If hover must lift the loaded vehicle, switch
    # m_drone_empty → m_drone_loaded.
    thrust_vtol_total = i.T_W_to * si.m_drone_empty * g0
    thrust_vtol_per_prop = thrust_vtol_total / i.n_props
    op_vtol = _solve_quiet(
        i.verbose,
        prop_solver.vtolsolve,
        i.csv_prop, thrust_vtol_per_prop, D_prop,
        i.vtol_rpm_init, i.vtol_thrust_tol, i.vtol_max_iter, i.vtol_rpm_step,
    )
    if op_vtol is None:
        raise RuntimeError("VTOL prop operating point did not converge.")
    n_vtol = op_vtol["RPM"] / 60.0
    Cp_vtol = op_vtol["Cp"]
    Ct_vtol = op_vtol["Ct"]
    P_shaft_vtol = i.n_props * _shaft_power_per_prop(Cp_vtol, s.rho, n_vtol, D_prop)
    P_elec_vtol = P_shaft_vtol / i.eff_motor

    # ----- Motor sizing from the worst-case (max-RPM) segment -----
    max_rpm = max(n_cruise, n_climb, n_vtol) * 60.0
    KV = max_rpm / (i.throttle_max * voltage_battery)
    throttle_cruise = (n_cruise * 60.0) / (KV * voltage_battery)
    throttle_climb = (n_climb * 60.0) / (KV * voltage_battery)
    throttle_vtol = (n_vtol * 60.0) / (KV * voltage_battery)
    motor_mass = motor_mass_from_kv(KV)
    total_motor_mass = motor_mass * i.n_props

    # ----- Energy & Battery -----
    E_cruise = P_elec_cruise * s.t_cruise
    E_climb = P_elec_climb * i.t_climb
    E_total = E_cruise + E_climb
    battery_mass = battery_mass_from_energy(E_total, i.n_cells, i.voltage_cell)
    battery_volume = battery_volume_from_energy(E_total, i.battery_density)

    return PropulsionResult(
        inputs=inputs,
        voltage_battery=voltage_battery,
        D_prop=D_prop,
        thrust_cruise_per_prop=thrust_cruise_per_prop,
        n_cruise=n_cruise,
        J_cruise=J_cruise,
        Cp_cruise=Cp_cruise,
        eff_cruise=eff_cruise,
        P_shaft_cruise=P_shaft_cruise,
        P_elec_cruise=P_elec_cruise,
        throttle_cruise=throttle_cruise,
        CL_climb=CL_climb,
        V_climb=V_climb,
        Cd_climb=Cd_climb,
        D_climb=D_climb,
        thrust_climb_per_prop=thrust_climb_per_prop,
        n_climb=n_climb,
        J_climb=J_climb,
        Cp_climb=Cp_climb,
        eff_climb=eff_climb,
        P_shaft_climb=P_shaft_climb,
        P_elec_climb=P_elec_climb,
        throttle_climb=throttle_climb,
        thrust_vtol_per_prop=thrust_vtol_per_prop,
        n_vtol=n_vtol,
        Cp_vtol=Cp_vtol,
        Ct_vtol=Ct_vtol,
        P_shaft_vtol=P_shaft_vtol,
        P_elec_vtol=P_elec_vtol,
        throttle_vtol=throttle_vtol,
        KV=KV,
        motor_mass=motor_mass,
        total_motor_mass=total_motor_mass,
        E_cruise=E_cruise,
        E_climb=E_climb,
        E_total=E_total,
        battery_mass=battery_mass,
        battery_volume=battery_volume,
    )


def summary(r: PropulsionResult) -> None:
    print("\n--- Propulsion: Cruise ---")
    print(f"  Thrust / prop        : {r.thrust_cruise_per_prop:.2f}  N")
    print(f"  RPM                  : {r.n_cruise * 60:.0f}")
    print(f"  J                    : {r.J_cruise:.4f}")
    print(f"  Cp                   : {r.Cp_cruise:.4f}")
    print(f"  Prop efficiency      : {r.eff_cruise:.3f}")
    print(f"  Shaft power (total)  : {r.P_shaft_cruise:.1f}  W")
    print(f"  Throttle             : {r.throttle_cruise:.3f}")

    print("\n--- Propulsion: Climb (best ROC) ---")
    print(f"  CL_climb             : {r.CL_climb:.3f}")
    print(f"  V_climb              : {r.V_climb:.2f}  m/s")
    print(f"  Drag at climb        : {r.D_climb:.2f}  N")
    print(f"  Thrust / prop        : {r.thrust_climb_per_prop:.2f}  N")
    print(f"  RPM                  : {r.n_climb * 60:.0f}")
    print(f"  J                    : {r.J_climb:.4f}")
    print(f"  Cp                   : {r.Cp_climb:.4f}")
    print(f"  Prop efficiency      : {r.eff_climb:.3f}")
    print(f"  Shaft power (total)  : {r.P_shaft_climb:.1f}  W")
    print(f"  Throttle             : {r.throttle_climb:.3f}")

    print("\n--- Propulsion: VTOL / hover ---")
    print(f"  Thrust / prop        : {r.thrust_vtol_per_prop:.2f}  N")
    print(f"  RPM                  : {r.n_vtol * 60:.0f}")
    print(f"  Cp                   : {r.Cp_vtol:.4f}")
    print(f"  Ct                   : {r.Ct_vtol:.4f}")
    print(f"  Shaft power (total)  : {r.P_shaft_vtol:.1f}  W")
    print(f"  Throttle             : {r.throttle_vtol:.3f}")

    print("\n--- Motor ---")
    print(f"  Propeller diameter   : {r.D_prop:.4f}  m")
    print(f"  KV                   : {r.KV:.1f}  RPM/V")
    print(f"  Motor mass (each)    : {r.motor_mass:.3f}  kg")
    print(f"  Total motor mass     : {r.total_motor_mass:.3f}  kg")

    print("\n--- Energy & Battery ---")
    print(f"  Cruise energy        : {r.E_cruise / 3600:.2f}  Wh")
    print(f"  Climb energy         : {r.E_climb / 3600:.2f}  Wh")
    print(f"  Total energy         : {r.E_total / 3600:.2f}  Wh")
    print(f"  Battery mass         : {r.battery_mass:.3f}  kg")
    print(f"  Battery volume       : {r.battery_volume * 1000:.2f}  L")


if __name__ == "__main__":
    from sizing.wing import SizingInputs
    summary(run(wing.run(SizingInputs())))
