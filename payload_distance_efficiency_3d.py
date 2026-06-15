"""3D payload/range efficiency map for the final drone design.

For each payload mass and mission distance, this script evaluates 2 through
6 drones using the converged final design. A bar is drawn only if at least one
drone count can:

  * carry the payload at 20 m/s without exceeding CLmax,
  * meet the cruise thrust demand, and
  * fly the requested distance with the available final-design battery.

The bar height is the best feasible fleet cruise energy per km. The bar color
shows which number of drones is most efficient for that payload/range cell.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from pipeline import config, loop
from sizing import wing


DRONE_COLORS = {
    2: "#4C78A8",
    3: "#F58518",
    4: "#54A24B",
    5: "#E45756",
    6: "#B279A2",
}


def _run_final_design(*, quiet: bool):
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return loop.run_pipeline(config)
    return loop.run_pipeline(config)


def _final_design_values(result) -> dict[str, float | int | str]:
    qS = result.sizing.q_cruise * result.sizing.Sw
    cruise_drag = result.cd_full_buildup * qS
    drivetrain_efficiency = (
        cruise_drag * result.sizing.inputs.V_cruise / result.propulsion.P_elec_cruise
        if result.propulsion.P_elec_cruise > 0.0 else float("nan")
    )
    cd_payload = result.sizing.inputs.Cd_payload * result.sizing.inputs.S_payload / (
        result.sizing.inputs.n_drones * result.sizing.Sw
    )
    cd_fixed = result.cd_full_buildup - result.sizing.k * result.sizing.CL**2 - cd_payload

    battery_nominal_wh = result.propulsion.E_total / 3600.0
    battery_usable_wh = battery_nominal_wh * result.propulsion.inputs.DoD
    climb_wh = result.propulsion.E_climb / 3600.0

    return {
        "source": "pipeline.loop.run_pipeline(config)",
        "airfoil": result.airfoil,
        "tail_airfoil": result.tail_airfoil,
        "payload_kg": result.sizing.inputs.m_payload,
        "n_drones": result.sizing.inputs.n_drones,
        "empty_mass_per_drone_kg": result.masses["total"],
        "battery_mass_kg": result.propulsion.battery_mass,
        "battery_nominal_wh_per_drone": battery_nominal_wh,
        "battery_usable_wh_per_drone": battery_usable_wh,
        "climb_wh_per_drone": climb_wh,
        "cruise_available_wh_per_drone": max(battery_usable_wh - climb_wh, 0.0),
        "battery_DoD": result.propulsion.inputs.DoD,
        "wing_span_m": result.sizing.inputs.b,
        "wing_area_m2": result.sizing.Sw,
        "aspect_ratio": result.sizing.inputs.AR,
        "tail_boom_length_m": result.sizing.L_boom,
        "horizontal_tail_area_m2": result.sizing.Sh,
        "cruise_speed_mps": result.sizing.inputs.V_cruise,
        "cruise_dynamic_pressure_pa": result.sizing.q_cruise,
        "CL_design": result.sizing.CL,
        "CLmax_wing": result.cl_max_wing,
        "CD_full_design": result.cd_full_buildup,
        "CD_fixed_no_wing_induced_no_payload": cd_fixed,
        "cruise_drag_per_drone_N": cruise_drag,
        "cruise_power_per_drone_W": result.propulsion.P_elec_cruise,
        "drivetrain_efficiency": drivetrain_efficiency,
        "max_thrust_per_drone_N": (
            result.propulsion.max_thrust_per_prop * result.propulsion.inputs.n_props
        ),
        "max_power_per_drone_W": result.propulsion.max_power_elec,
    }


def _load_or_run_final_design(args) -> dict[str, float | int | str]:
    if args.reuse_final_design and args.final_design.exists():
        with args.final_design.open() as f:
            values = json.load(f)
        required = {
            "battery_usable_wh_per_drone",
            "climb_wh_per_drone",
            "cruise_available_wh_per_drone",
        }
        if required.issubset(values):
            return values

    print("Running full sizing pipeline to get the converged final design...")
    result = _run_final_design(quiet=not args.verbose_pipeline)
    values = _final_design_values(result)
    with args.final_design.open("w") as f:
        json.dump(values, f, indent=2)
        f.write("\n")
    return values


def _mission_for_drone_count(
    final_design: dict[str, float | int | str],
    *,
    n_drones: int,
    payload_kg: float,
    distance_km: float,
) -> dict[str, float | int | bool | str]:
    sizing_inputs = dataclasses.replace(
        config.SIZING,
        m_payload=float(payload_kg),
        n_drones=n_drones,
        m_drone_empty=float(final_design["empty_mass_per_drone_kg"]),
        b=float(final_design["wing_span_m"]),
        AR=float(final_design["aspect_ratio"]),
        L_boom=float(final_design["tail_boom_length_m"]),
        Sh=float(final_design["horizontal_tail_area_m2"]),
    )
    sizing = wing.run(
        sizing_inputs,
        c_aileron_to_c_wing=config.CONTROL_SURFACE.c_aileron_to_c_wing,
    )

    cd_payload = sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
        sizing.inputs.n_drones * sizing.Sw
    )
    cd_total = (
        float(final_design["CD_fixed_no_wing_induced_no_payload"])
        + sizing.k * sizing.CL**2
        + cd_payload
    )
    drag_per_drone = cd_total * sizing.q_cruise * sizing.Sw
    power_per_drone = (
        drag_per_drone
        * float(final_design["cruise_speed_mps"])
        / float(final_design["drivetrain_efficiency"])
    )
    cruise_wh_per_drone = power_per_drone * (distance_km * 1000.0) / (
        float(final_design["cruise_speed_mps"]) * 3600.0
    )
    wh_per_km_fleet = n_drones * drag_per_drone / (
        3.6 * float(final_design["drivetrain_efficiency"])
    )

    reasons: list[str] = []
    if sizing.CL > float(final_design["CLmax_wing"]):
        reasons.append(
            f"CL {sizing.CL:.3f} > CLmax {float(final_design['CLmax_wing']):.3f}"
        )
    if drag_per_drone > float(final_design["max_thrust_per_drone_N"]):
        reasons.append(
            "thrust "
            f"{drag_per_drone:.1f} N > {float(final_design['max_thrust_per_drone_N']):.1f} N"
        )
    if power_per_drone > float(final_design["max_power_per_drone_W"]):
        reasons.append(
            "power "
            f"{power_per_drone:.0f} W > {float(final_design['max_power_per_drone_W']):.0f} W"
        )
    if cruise_wh_per_drone > float(final_design["cruise_available_wh_per_drone"]):
        reasons.append(
            "battery "
            f"{cruise_wh_per_drone:.1f} Wh > "
            f"{float(final_design['cruise_available_wh_per_drone']):.1f} Wh"
        )

    return {
        "n_drones": n_drones,
        "payload_kg": payload_kg,
        "distance_km": distance_km,
        "fleet_Wh_per_km": wh_per_km_fleet,
        "cruise_Wh_per_drone": cruise_wh_per_drone,
        "available_cruise_Wh_per_drone": float(final_design["cruise_available_wh_per_drone"]),
        "CL": float(sizing.CL),
        "CD_total": float(cd_total),
        "thrust_required_per_drone_N": float(drag_per_drone),
        "power_required_per_drone_W": float(power_per_drone),
        "feasible": not reasons,
        "reason": "; ".join(reasons),
    }


def _best_grid(
    final_design: dict[str, float | int | str],
    payloads_kg: np.ndarray,
    distances_km: np.ndarray,
):
    all_rows: list[dict[str, float | int | bool | str]] = []
    best_rows: list[dict[str, float | int | bool | str]] = []

    for payload_kg in payloads_kg:
        for distance_km in distances_km:
            candidates = [
                _mission_for_drone_count(
                    final_design,
                    n_drones=n,
                    payload_kg=float(payload_kg),
                    distance_km=float(distance_km),
                )
                for n in (2, 3, 4, 5, 6)
            ]
            all_rows.extend(candidates)
            feasible = [r for r in candidates if r["feasible"]]
            if feasible:
                best = min(feasible, key=lambda r: float(r["fleet_Wh_per_km"]))
                best_rows.append({
                    "payload_kg": float(payload_kg),
                    "distance_km": float(distance_km),
                    "best_n_drones": int(best["n_drones"]),
                    "best_fleet_Wh_per_km": float(best["fleet_Wh_per_km"]),
                    "feasible": True,
                    "reason": "",
                })
            else:
                best_rows.append({
                    "payload_kg": float(payload_kg),
                    "distance_km": float(distance_km),
                    "best_n_drones": 0,
                    "best_fleet_Wh_per_km": float("nan"),
                    "feasible": False,
                    "reason": "no drone count can carry payload and fly range",
                })
    return best_rows, all_rows


def _write_csv(rows, output: Path) -> None:
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_3d(
    best_rows: list[dict[str, float | int | bool | str]],
    *,
    payload_step: float,
    distance_step: float,
    output: Path,
    show: bool,
) -> None:
    feasible = [r for r in best_rows if r["feasible"]]
    if not feasible:
        raise RuntimeError("No feasible payload/range cells to plot.")

    x = np.array([float(r["payload_kg"]) for r in feasible])
    y = np.array([float(r["distance_km"]) for r in feasible])
    z = np.zeros_like(x)
    dz = np.array([float(r["best_fleet_Wh_per_km"]) for r in feasible])
    colors = [DRONE_COLORS[int(r["best_n_drones"])] for r in feasible]

    dx = max(payload_step * 0.82, 0.2)
    dy = max(distance_step * 0.82, 0.2)

    fig = plt.figure(figsize=(10.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(x - dx / 2.0, y - dy / 2.0, z, dx, dy, dz, color=colors, shade=True, alpha=0.92)

    ax.set_xlabel("Payload mass [kg]")
    ax.set_ylabel("Distance [km]")
    ax.set_zlabel("Best fleet energy [Wh/km]")
    ax.set_title("Most Efficient Feasible Drone Count")
    ax.view_init(elev=28, azim=-58)

    handles = [
        patches.Patch(color=DRONE_COLORS[n], label=f"{n} drones")
        for n in sorted(DRONE_COLORS)
    ]
    ax.legend(handles=handles, title="Best fleet size", loc="upper left")

    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("payload_distance_efficiency_3d.svg"))
    parser.add_argument("--best-csv", type=Path, default=Path("payload_distance_efficiency_best.csv"))
    parser.add_argument("--all-csv", type=Path, default=Path("payload_distance_efficiency_all.csv"))
    parser.add_argument("--final-design", type=Path, default=Path("final_design_values_3d.json"))
    parser.add_argument("--reuse-final-design", action="store_true")
    parser.add_argument("--payload-max", type=float, default=120.0)
    parser.add_argument("--payload-step", type=float, default=5.0)
    parser.add_argument("--distance-max", type=float, default=40.0)
    parser.add_argument("--distance-step", type=float, default=2.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose-pipeline", action="store_true")
    args = parser.parse_args()

    final_design = _load_or_run_final_design(args)
    payloads = np.arange(0.0, args.payload_max + 0.5 * args.payload_step, args.payload_step)
    distances = np.arange(args.distance_step, args.distance_max + 0.5 * args.distance_step, args.distance_step)

    best_rows, all_rows = _best_grid(final_design, payloads, distances)
    _write_csv(best_rows, args.best_csv)
    _write_csv(all_rows, args.all_csv)
    _plot_3d(
        best_rows,
        payload_step=args.payload_step,
        distance_step=args.distance_step,
        output=args.output,
        show=args.show,
    )

    feasible = [r for r in best_rows if r["feasible"]]
    print("3D payload/range efficiency map complete")
    print(f"  feasible cells: {len(feasible)} / {len(best_rows)}")
    print(
        f"  available cruise battery per drone: "
        f"{float(final_design['cruise_available_wh_per_drone']):.1f} Wh"
    )
    print(f"Saved plot: {args.output}")
    print(f"Saved best-grid CSV: {args.best_csv}")
    print(f"Saved all-candidates CSV: {args.all_csv}")
    print(f"Saved final design: {args.final_design}")


if __name__ == "__main__":
    main()
