"""One-drone-failure cruise and emergency range check.

Runs the converged design once, then checks whether two active drones can
continue cruise while carrying the shared payload plus the failed drone mass.
The failed drone is treated as payload weight and contributes only CD0 drag.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from pipeline import config, loop
from pipeline.sensitivity import clone_config
from propulsion import propeller_solver as prop_solver
from propulsion.sizing import _diameter_from_csv_name, _shaft_power_per_prop
from sizing.wing import g0


@dataclass
class FailurePoint:
    V: float
    CL: float
    drag_total: float
    thrust_per_prop: float
    power_total: float
    wh_per_km: float
    feasible: bool
    message: str = ""


def _solve_prop_power(result, thrust_per_prop: float, V: float) -> tuple[float, str]:
    p = result.propulsion
    i = p.inputs
    D_prop = p.D_prop if p.D_prop is not None else _diameter_from_csv_name(i.csv_prop)
    with contextlib.redirect_stdout(io.StringIO()):
        op = prop_solver.solve(
            i.csv_prop,
            thrust_per_prop,
            V,
            D_prop,
            max(p.n_cruise * 60.0, 1000.0),
            i.rpm_tol,
            i.thrust_tol,
            i.max_iter,
            i.rpm_step,
        )
    if op is None:
        return math.inf, "prop operating point did not converge"

    n_rps = op["RPM"] / 60.0
    p_shaft_per_prop = _shaft_power_per_prop(op["Cp"], result.sizing.rho, n_rps, D_prop)
    n_active = result.sizing.inputs.n_drones - 1
    p_elec_total = n_active * i.n_props * p_shaft_per_prop / i.eff_motor
    return float(p_elec_total), ""


def _failure_point(result, V: float) -> FailurePoint:
    s = result.sizing
    n_total = s.inputs.n_drones
    n_active = n_total - 1
    if n_active < 1:
        return FailurePoint(V, math.inf, math.inf, math.inf, math.inf, math.inf, False, "need at least two drones")

    m_active = result.masses["total"]
    m_dead = m_active
    m_payload_failed = s.inputs.m_payload + m_dead
    q = 0.5 * s.rho * V**2
    W_total = (n_active * m_active + m_payload_failed) * g0
    CL = W_total / (n_active * q * s.Sw)

    cd0_active = result.drag.CD0
    cd0_dead = result.drag.CD0
    cdi_active = CL**2 / (np.pi * s.e * s.inputs.AR)
    D_active = n_active * q * s.Sw * (cd0_active + cdi_active)
    D_dead = q * s.Sw * cd0_dead
    D_payload = q * s.inputs.Cd_payload * s.inputs.S_payload
    D_total = D_active + D_dead + D_payload

    thrust_per_prop = D_total / (n_active * result.propulsion.inputs.n_props)
    if CL > result.cl_max_wing:
        return FailurePoint(V, CL, D_total, thrust_per_prop, math.inf, math.inf, False, "wing CL exceeds CLmax")
    if thrust_per_prop > result.propulsion.max_thrust_per_prop:
        return FailurePoint(V, CL, D_total, thrust_per_prop, math.inf, math.inf, False, "thrust limit")

    power_total, prop_message = _solve_prop_power(result, thrust_per_prop, V)
    if not np.isfinite(power_total):
        return FailurePoint(V, CL, D_total, thrust_per_prop, power_total, math.inf, False, prop_message)
    if power_total > n_active * result.propulsion.max_power_elec:
        return FailurePoint(V, CL, D_total, thrust_per_prop, power_total, math.inf, False, "power limit")

    wh_per_km = power_total / V / 3.6
    return FailurePoint(V, CL, D_total, thrust_per_prop, power_total, wh_per_km, True)


def evaluate_failure_envelope(result, *, v_max: float, samples: int) -> tuple[list[FailurePoint], FailurePoint | None]:
    s = result.sizing
    n_active = s.inputs.n_drones - 1
    m_active = result.masses["total"]
    W_total = (n_active * m_active + s.inputs.m_payload + m_active) * g0
    v_stall_failure = math.sqrt(W_total / (n_active * 0.5 * s.rho * s.Sw * result.cl_max_wing))
    v_lo = max(0.95 * v_stall_failure, 5.0)
    speeds = np.linspace(v_lo, v_max, samples)
    points = [_failure_point(result, float(V)) for V in speeds]
    feasible = [p for p in points if p.feasible]
    best = min(feasible, key=lambda p: p.wh_per_km) if feasible else None
    return points, best


def plot_range(result, best: FailurePoint, *, output: str | None, show: bool) -> None:
    capacity_Wh = result.propulsion.E_total / 3600.0
    reserve_Wh = (1.0 - result.propulsion.inputs.DoD) * capacity_Wh
    climb_Wh = result.propulsion.E_climb / 3600.0
    cruise_Wh = result.propulsion.E_cruise / 3600.0
    mission_km = result.sizing.inputs.R / 1000.0
    progress_km = np.linspace(0.0, mission_km, 200)
    remaining_Wh = capacity_Wh - climb_Wh - cruise_Wh * (progress_km / mission_km)
    remaining_Wh = np.maximum(remaining_Wh, 0.0)
    ranges_km = remaining_Wh / best.wh_per_km
    reserve_range_km = reserve_Wh / best.wh_per_km

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(progress_km, ranges_km, linewidth=2)
    ax.axhline(reserve_range_km, color="C3", linestyle="--", linewidth=1.2,
               label=f"20% emergency reserve range: {reserve_range_km:.1f} km")
    ax.scatter([mission_km], [reserve_range_km], color="C3", zorder=5)
    ax.text(
        0.98 * mission_km,
        reserve_range_km,
        f"{reserve_range_km:.2f} km",
        color="C3",
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("Three-drone cruise progress [km]")
    ax.set_ylabel("Two-drone failure-cruise range [km]")
    ax.set_title("One-drone-failure recovery range vs nominal cruise progress")
    ax.set_xlim(0.0, mission_km)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-max", type=float, default=25.0, help="Maximum failure-cruise speed [m/s]")
    parser.add_argument("--samples", type=int, default=80, help="Number of speed samples")
    parser.add_argument("--output", default="one_drone_failure_range.svg", help="Output plot path")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    parser.add_argument("--verbose-pipeline", action="store_true", help="Show full sizing pipeline output")
    args = parser.parse_args()

    cfg = clone_config(sizing=config.SIZING, propulsion=config.PROPULSION)
    if args.verbose_pipeline:
        result = loop.run_pipeline(cfg)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = loop.run_pipeline(cfg)

    points, best = evaluate_failure_envelope(result, v_max=args.v_max, samples=args.samples)
    if best is None:
        print("No feasible two-drone failure-cruise point found.")
        for point in points:
            if point.message:
                print(f"  V={point.V:5.2f} m/s  CL={point.CL:6.3f}  infeasible: {point.message}")
        raise SystemExit(1)

    capacity_Wh = result.propulsion.E_total / 3600.0
    reserve_Wh = (1.0 - result.propulsion.inputs.DoD) * capacity_Wh
    start_cruise_remaining_Wh = capacity_Wh - result.propulsion.E_climb / 3600.0
    mission_km = result.sizing.inputs.R / 1000.0
    print("One-drone-failure cruise check")
    print(f"  Assumption: failed drone mass is added to payload; failed drone adds CD0 drag only.")
    print(f"  Best feasible speed        : {best.V:.2f} m/s")
    print(f"  Required active-drone CL   : {best.CL:.3f} / CLmax {result.cl_max_wing:.3f}")
    print(f"  Total drag                 : {best.drag_total:.2f} N")
    print(f"  Thrust per active prop     : {best.thrust_per_prop:.2f} N / limit {result.propulsion.max_thrust_per_prop:.2f} N")
    print(f"  Total active electrical P  : {best.power_total:.1f} W / limit {(result.sizing.inputs.n_drones - 1) * result.propulsion.max_power_elec:.1f} W")
    print(f"  Energy per km              : {best.wh_per_km:.2f} Wh/km")
    print(f"  Emergency reserve energy   : {reserve_Wh:.2f} Wh ({100.0 * (1.0 - result.propulsion.inputs.DoD):.0f}% of pack)")
    print(f"  Range if failure at 0 km   : {start_cruise_remaining_Wh / best.wh_per_km:.2f} km")
    print(f"  Emergency reserve range    : {reserve_Wh / best.wh_per_km:.2f} km")
    print(f"  Range if failure at {mission_km:.0f} km  : {reserve_Wh / best.wh_per_km:.2f} km")

    plot_range(result, best, output=args.output, show=args.show)
    print(f"  Plot saved to              : {args.output}")


if __name__ == "__main__":
    main()
