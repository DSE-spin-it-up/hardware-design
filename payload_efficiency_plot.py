"""Plot final-design cruise energy per km versus payload mass.

The final design is found by running the full sizing pipeline once from
``config.yaml``. That converged design is then frozen while payload mass is
swept from 0 to 120 kg for 2, 3, 4, 5, and 6 drones.
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

from pipeline import config, loop
from sizing import wing


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

    return {
        "source": "pipeline.loop.run_pipeline(config)",
        "airfoil": result.airfoil,
        "tail_airfoil": result.tail_airfoil,
        "payload_kg": result.sizing.inputs.m_payload,
        "n_drones": result.sizing.inputs.n_drones,
        "empty_mass_per_drone_kg": result.masses["total"],
        "battery_mass_kg": result.propulsion.battery_mass,
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
        "n_props_per_drone": result.propulsion.inputs.n_props,
        "max_power_per_drone_W": result.propulsion.max_power_elec,
    }


def _save_final_design(values: dict[str, float | int | str], output: Path) -> None:
    with output.open("w") as f:
        json.dump(values, f, indent=2)
        f.write("\n")


def _payload_reason(
    *,
    cl_required: float,
    cl_max: float,
    thrust_required: float,
    thrust_available: float,
) -> str:
    reasons: list[str] = []
    if cl_required > cl_max:
        reasons.append(
            f"drones cannot carry payload: CL {cl_required:.3f} > CLmax {cl_max:.3f}"
        )
    if thrust_required > thrust_available:
        reasons.append(
            "drones cannot carry payload: "
            f"thrust {thrust_required:.1f} N > available {thrust_available:.1f} N"
        )
    return "; ".join(reasons)


def _sweep(final_design: dict[str, float | int | str], payloads_kg: np.ndarray):
    rows: list[dict[str, float | int | bool | str]] = []
    base_inputs = config.SIZING
    drone_counts = (2, 3, 4, 5, 6)

    for n_drones in drone_counts:
        already_limited = False
        for payload_kg in payloads_kg:
            sizing_inputs = dataclasses.replace(
                base_inputs,
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
            reason = _payload_reason(
                cl_required=float(sizing.CL),
                cl_max=float(final_design["CLmax_wing"]),
                thrust_required=float(drag_per_drone),
                thrust_available=float(final_design["max_thrust_per_drone_N"]),
            )
            feasible = reason == "" and not already_limited
            if reason:
                already_limited = True
            wh_per_km = (
                n_drones * drag_per_drone
                / (3.6 * float(final_design["drivetrain_efficiency"]))
                if feasible else float("nan")
            )
            rows.append(
                {
                    "n_drones": n_drones,
                    "payload_kg": float(payload_kg),
                    "fleet_Wh_per_km": wh_per_km,
                    "CL": float(sizing.CL),
                    "CLmax_wing": float(final_design["CLmax_wing"]),
                    "CD_total": float(cd_total),
                    "thrust_required_per_drone_N": float(drag_per_drone),
                    "thrust_available_per_drone_N": float(final_design["max_thrust_per_drone_N"]),
                    "feasible": feasible,
                    "reason": reason if reason else ("limited by lower payload" if already_limited else ""),
                }
            )
    return rows, drone_counts


def _plot(rows, drone_counts: tuple[int, ...], output: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for n_drones in drone_counts:
        group = [r for r in rows if r["n_drones"] == n_drones]
        payload = np.array([float(r["payload_kg"]) for r in group])
        wh_per_km = np.array([float(r["fleet_Wh_per_km"]) for r in group])
        feasible = np.array([bool(r["feasible"]) for r in group])
        ax.plot(payload, wh_per_km, linewidth=2.0, label=f"{n_drones} drones")

        if np.any(~feasible):
            first_bad = group[int(np.argmax(~feasible))]
            bad_payload = float(first_bad["payload_kg"])
            last_ok_payloads = payload[feasible]
            last_ok_energy = wh_per_km[feasible]
            if last_ok_payloads.size and np.isfinite(last_ok_energy[-1]):
                ax.scatter(last_ok_payloads[-1], last_ok_energy[-1], s=28)
                ax.annotate(
                    f"{n_drones} drones cannot carry >{bad_payload:.0f} kg",
                    xy=(last_ok_payloads[-1], last_ok_energy[-1]),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_xlabel("Payload mass [kg]")
    ax.set_ylabel("Cruise fleet energy per distance [Wh/km]")
    ax.set_title("Final Design Payload Transport Efficiency")
    ax.set_xlim(0.0, 200.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Fleet size")
    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def _write_csv(rows, output: Path) -> None:
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_limits(rows, drone_counts: tuple[int, ...]) -> None:
    for n_drones in drone_counts:
        group = [r for r in rows if r["n_drones"] == n_drones]
        feasible = [r for r in group if r["feasible"]]
        first_bad = next((r for r in group if not r["feasible"]), None)
        if feasible:
            print(
                f"  {n_drones} drones: curve reaches "
                f"{float(feasible[-1]['payload_kg']):.0f} kg"
            )
        if first_bad is not None:
            print(
                f"    cannot carry {float(first_bad['payload_kg']):.0f} kg: "
                f"{first_bad['reason']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("payload_efficiency.svg"))
    parser.add_argument("--csv", type=Path, default=Path("payload_efficiency.csv"))
    parser.add_argument("--final-design", type=Path, default=Path("final_design_values.json"))
    parser.add_argument("--payload-max", type=float, default=200.0)
    parser.add_argument("--payload-step", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose-pipeline", action="store_true")
    args = parser.parse_args()

    print("Running full sizing pipeline to get the converged final design...")
    result = _run_final_design(quiet=not args.verbose_pipeline)
    final_design = _final_design_values(result)
    _save_final_design(final_design, args.final_design)

    payloads = np.arange(0.0, args.payload_max + 0.5 * args.payload_step, args.payload_step)
    rows, drone_counts = _sweep(final_design, payloads)
    _write_csv(rows, args.csv)
    _plot(rows, drone_counts, args.output, args.show)

    print("Final design saved from converged pipeline:")
    print(f"  mass per drone: {final_design['empty_mass_per_drone_kg']:.3f} kg")
    print(f"  wing area: {final_design['wing_area_m2']:.4f} m^2")
    print(f"  CLmax wing: {final_design['CLmax_wing']:.3f}")
    print(f"  max thrust per drone: {final_design['max_thrust_per_drone_N']:.1f} N")
    print("Payload limits:")
    _print_limits(rows, drone_counts)
    print(f"Saved final design: {args.final_design}")
    print(f"Saved plot: {args.output}")
    print(f"Saved data: {args.csv}")


if __name__ == "__main__":
    main()
