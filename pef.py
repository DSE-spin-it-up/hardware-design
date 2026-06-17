"""Plot final-design best fleet size versus payload mass and range.

The final design is loaded from ``final_design_values.json`` and then frozen
while payload mass and range are swept for 3, 4, 5, and 6 drones.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from pipeline import config
from sizing import wing
from sizing.wing import g0


FinalDesign = dict[str, float | int | str]
PLOT_PAYLOAD_OFFSET_KG = 10.0
PLOT_PAYLOAD_X_MAX_KG = 110.0
PLOT_PAYLOAD_TICKS_KG = np.arange(0.0, PLOT_PAYLOAD_X_MAX_KG + 1.0, 10.0)
DRONE_COLORS = {
    3: "#2c7fb8",
    4: "#41ab5d",
    5: "#fdae61",
    6: "#d7191c",
}


def _load_final_design(path: Path) -> FinalDesign:
    with path.open() as f:
        values = json.load(f)

    required = (
        "empty_mass_per_drone_kg",
        "battery_mass_kg",
        "payload_kg",
        "n_drones",
        "wing_span_m",
        "aspect_ratio",
        "tail_boom_length_m",
        "horizontal_tail_area_m2",
        "CLmax_wing",
        "CD_fixed_no_wing_induced_no_payload",
        "drivetrain_efficiency",
        "max_thrust_per_drone_N",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(
            f"{path} is missing required final design value(s): {', '.join(missing)}"
        )
    return values


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


def _battery_capacity_wh_from_mass(
    battery_mass_kg: float,
    *,
    n_cells: int,
    voltage_cell: float,
) -> float:
    """Invert the 6S battery mass fit used by ``propulsion.sizing``."""
    if n_cells != 6:
        raise NotImplementedError(f"battery capacity fit not available for n_cells={n_cells}")
    a, b = 0.3988, 0.8810
    mass_g = battery_mass_kg * 1000.0
    capacity_mah = (mass_g / a) ** (1.0 / b)
    return capacity_mah * n_cells * voltage_cell / 1000.0


def _climb_energy_per_drone_Wh(
    sizing: wing.SizingResult,
    *,
    cd_fixed_no_wing_induced_no_payload: float,
    drivetrain_efficiency: float,
) -> float:
    """Estimate per-drone climb energy for the dense payload/range sweep."""
    cl_climb = np.sqrt(3.0 * cd_fixed_no_wing_induced_no_payload / sizing.k)
    v_climb = np.sqrt(
        2.0 * sizing.m_drone_loaded * g0 / (sizing.rho * sizing.Sw * cl_climb)
    )
    cd_payload = (
        sizing.inputs.Cd_payload
        * sizing.inputs.S_payload
        / (sizing.inputs.n_drones * sizing.Sw)
        if sizing.inputs.m_payload > 0.0
        else 0.0
    )
    cd_climb = cd_fixed_no_wing_induced_no_payload + sizing.k * cl_climb**2 + cd_payload
    drag_climb = 0.5 * sizing.rho * v_climb**2 * sizing.Sw * cd_climb
    climb_power_W = (
        drag_climb * v_climb
        + config.PROPULSION.climb_rate * sizing.m_drone_loaded * g0
    ) / drivetrain_efficiency
    return climb_power_W * config.PROPULSION.t_climb / 3600.0


def _sweep(final_design: FinalDesign, payloads_kg: np.ndarray):
    rows: list[dict[str, float | int | bool | str]] = []
    base_inputs = config.SIZING
    drone_counts = (3, 4, 5, 6)
    battery_capacity_per_drone_Wh = _battery_capacity_wh_from_mass(
        float(final_design["battery_mass_kg"]),
        n_cells=config.PROPULSION.n_cells,
        voltage_cell=config.PROPULSION.voltage_cell,
    )
    usable_battery_fraction = float(config.PROPULSION.DoD)
    usable_battery_per_drone_Wh = battery_capacity_per_drone_Wh * usable_battery_fraction

    for n_drones in drone_counts:
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

            cd_payload = (
                sizing.inputs.Cd_payload * sizing.inputs.S_payload / (
                    sizing.inputs.n_drones * sizing.Sw
                )
                if payload_kg > 0.0 else 0.0
            )
            cd_total = (
                float(final_design["CD_fixed_no_wing_induced_no_payload"])
                + sizing.k * sizing.CL**2
                + cd_payload
            )
            drag_per_drone = cd_total * sizing.q_cruise * sizing.Sw
            climb_energy_per_drone_Wh = _climb_energy_per_drone_Wh(
                sizing,
                cd_fixed_no_wing_induced_no_payload=float(
                    final_design["CD_fixed_no_wing_induced_no_payload"]
                ),
                drivetrain_efficiency=float(final_design["drivetrain_efficiency"]),
            )
            fleet_climb_energy_Wh = n_drones * climb_energy_per_drone_Wh
            payload_per_drone_kg = float(payload_kg) / n_drones
            reason = _payload_reason(
                cl_required=float(sizing.CL),
                cl_max=float(final_design["CLmax_wing"]),
                thrust_required=float(drag_per_drone),
                thrust_available=float(final_design["max_thrust_per_drone_N"]),
            )
            feasible = reason == ""

            unconstrained_wh_per_km = (
                n_drones * drag_per_drone
                / (3.6 * float(final_design["drivetrain_efficiency"]))
            )
            wh_per_km = unconstrained_wh_per_km if feasible else float("nan")
            full_fleet_battery_Wh = n_drones * battery_capacity_per_drone_Wh
            usable_fleet_battery_Wh = n_drones * usable_battery_per_drone_Wh
            usable_cruise_battery_Wh = usable_fleet_battery_Wh - fleet_climb_energy_Wh
            max_range_km = (
                usable_cruise_battery_Wh / wh_per_km
                if feasible and np.isfinite(wh_per_km) and wh_per_km > 0.0
                and usable_cruise_battery_Wh > 0.0
                else float("nan")
            )
            rows.append(
                {
                    "n_drones": n_drones,
                    "payload_kg": float(payload_kg),
                    "payload_per_drone_kg": payload_per_drone_kg,
                    "total_system_Wh_per_km": wh_per_km,
                    "unconstrained_total_system_Wh_per_km": unconstrained_wh_per_km,
                    "climb_energy_per_drone_Wh": climb_energy_per_drone_Wh,
                    "total_system_climb_energy_Wh": fleet_climb_energy_Wh,
                    "battery_usable_fraction": usable_battery_fraction,
                    "total_system_battery_Wh": full_fleet_battery_Wh,
                    "usable_system_battery_Wh": usable_fleet_battery_Wh,
                    "usable_cruise_battery_Wh": (
                        usable_cruise_battery_Wh if feasible else float("nan")
                    ),
                    "battery_limited_range_km": max_range_km,
                    "CL": float(sizing.CL),
                    "CLmax_wing": float(final_design["CLmax_wing"]),
                    "CD_total": float(cd_total),
                    "thrust_required_per_drone_N": float(drag_per_drone),
                    "thrust_available_per_drone_N": float(final_design["max_thrust_per_drone_N"]),
                    "feasible": feasible,
                    "reason": reason,
                }
            )
    return rows, drone_counts


def _best_grid(rows, ranges_km: np.ndarray):
    payloads = np.array(sorted({float(r["payload_kg"]) for r in rows}))
    grid = np.zeros((len(ranges_km), len(payloads)), dtype=float)
    grid_rows: list[dict[str, float | int | bool | str]] = []
    rows_by_payload = {payload: [] for payload in payloads}

    for row in rows:
        rows_by_payload[float(row["payload_kg"])].append(row)

    for x_idx, payload in enumerate(payloads):
        feasible_options = [
            option for option in rows_by_payload[float(payload)]
            if (
                bool(option["feasible"])
                and np.isfinite(float(option["total_system_Wh_per_km"]))
                and np.isfinite(float(option["battery_limited_range_km"]))
            )
        ]

        for y_idx, range_km in enumerate(ranges_km):
            range_feasible_options = [
                option for option in feasible_options
                if float(option["battery_limited_range_km"]) >= float(range_km)
            ]
            best = (
                min(range_feasible_options, key=lambda r: float(r["total_system_Wh_per_km"]))
                if range_feasible_options
                else None
            )
            if best is not None:
                best_n = int(best["n_drones"])
                grid[y_idx, x_idx] = best_n
                grid_rows.append(
                    {
                        "payload_kg": float(payload),
                        "range_km": float(range_km),
                        "best_n_drones": best_n,
                        "total_system_Wh_per_km": float(best["total_system_Wh_per_km"]),
                        "total_system_climb_energy_Wh": float(
                            best["total_system_climb_energy_Wh"]
                        ),
                        "total_system_energy_required_Wh": (
                            float(best["total_system_climb_energy_Wh"])
                            +
                            float(best["total_system_Wh_per_km"]) * float(range_km)
                        ),
                        "battery_usable_fraction": float(best["battery_usable_fraction"]),
                        "usable_system_battery_Wh": float(best["usable_system_battery_Wh"]),
                        "usable_cruise_battery_Wh": float(best["usable_cruise_battery_Wh"]),
                        "battery_limited_range_km": float(best["battery_limited_range_km"]),
                        "feasible": True,
                        "reason": "",
                    }
                )
            else:
                if feasible_options:
                    longest_range_option = max(
                        feasible_options,
                        key=lambda r: float(r["battery_limited_range_km"]),
                    )
                    best_n = int(longest_range_option["n_drones"])
                    wh_per_km = float(longest_range_option["total_system_Wh_per_km"])
                    climb_energy_Wh = float(longest_range_option["total_system_climb_energy_Wh"])
                    energy_required_Wh = climb_energy_Wh + wh_per_km * float(range_km)
                    usable_battery_Wh = float(longest_range_option["usable_system_battery_Wh"])
                    usable_cruise_battery_Wh = float(
                        longest_range_option["usable_cruise_battery_Wh"]
                    )
                    battery_limited_range_km = float(
                        longest_range_option["battery_limited_range_km"]
                    )
                    battery_usable_fraction = float(
                        longest_range_option["battery_usable_fraction"]
                    )
                    reason = (
                        f"longest-range drone count uses {battery_usable_fraction:.0%} "
                        "battery after climb and cannot fly this range; "
                        f"needs {energy_required_Wh:.1f} Wh > usable {usable_battery_Wh:.1f} Wh"
                    )
                else:
                    best_n = 0
                    wh_per_km = float("nan")
                    climb_energy_Wh = float("nan")
                    energy_required_Wh = float("nan")
                    usable_battery_Wh = float("nan")
                    usable_cruise_battery_Wh = float("nan")
                    battery_limited_range_km = float("nan")
                    battery_usable_fraction = float(config.PROPULSION.DoD)
                    reason = "no drone count can carry payload"
                grid_rows.append(
                    {
                        "payload_kg": float(payload),
                        "range_km": float(range_km),
                        "best_n_drones": best_n,
                        "total_system_Wh_per_km": wh_per_km,
                        "total_system_climb_energy_Wh": climb_energy_Wh,
                        "total_system_energy_required_Wh": energy_required_Wh,
                        "battery_usable_fraction": battery_usable_fraction,
                        "usable_system_battery_Wh": usable_battery_Wh,
                        "usable_cruise_battery_Wh": usable_cruise_battery_Wh,
                        "battery_limited_range_km": battery_limited_range_km,
                        "feasible": False,
                        "reason": reason,
                    }
                )

    return payloads, grid, grid_rows


def _plot(
    payloads: np.ndarray,
    ranges_km: np.ndarray,
    grid: np.ndarray,
    rows: list[dict[str, float | int | bool | str]],
    drone_counts: tuple[int, ...],
    payload_max: float,
    range_max: float,
    output: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    fill_colors = ["#eeeeee", *[DRONE_COLORS[n] for n in drone_counts]]
    ax.set_facecolor(fill_colors[0])
    ax.set_axisbelow(True)
    ax.grid(True, color="#b8b8b8", linewidth=0.7, alpha=0.55)

    plot_payloads = payloads - PLOT_PAYLOAD_OFFSET_KG
    envelopes: dict[int, dict[str, np.ndarray]] = {}
    for n_drones in drone_counts:
        is_best = grid == n_drones
        valid = np.any(is_best, axis=0)
        bottom = np.full(plot_payloads.shape, np.nan, dtype=float)
        top = np.full(plot_payloads.shape, np.nan, dtype=float)
        for payload_idx in np.flatnonzero(valid):
            selected_ranges = ranges_km[is_best[:, payload_idx]]
            bottom[payload_idx] = np.min(selected_ranges)
            top[payload_idx] = np.max(selected_ranges)

        envelopes[n_drones] = {
            "payload": plot_payloads,
            "bottom": bottom,
            "top": top,
            "valid": valid,
        }

    for idx, n_drones in enumerate(drone_counts):
        line_zorder = 10 + len(drone_counts) - idx
        envelope = envelopes[n_drones]
        payload = envelope["payload"]
        bottom = envelope["bottom"]
        top = envelope["top"]
        valid = envelope["valid"].astype(bool)

        ax.fill_between(
            payload,
            bottom,
            top,
            where=valid,
            color=DRONE_COLORS[n_drones],
            alpha=0.55,
            interpolate=True,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            payload,
            np.where(valid, top, np.nan),
            color=DRONE_COLORS[n_drones],
            linewidth=2.8,
            zorder=line_zorder,
        )
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size:
            last_idx = int(valid_idx[-1])
            ax.vlines(
                payload[last_idx],
                bottom[last_idx],
                top[last_idx],
                color=DRONE_COLORS[n_drones],
                linewidth=2.8,
                zorder=line_zorder,
            )

    legend_handles = [
        Patch(facecolor=fill_colors[0], edgecolor="none", label="not feasible"),
        Line2D([0], [0], color=DRONE_COLORS[3], linewidth=2.8, label="3 drones"),
        Line2D([0], [0], color=DRONE_COLORS[4], linewidth=2.8, label="4 drones"),
        Line2D([0], [0], color=DRONE_COLORS[5], linewidth=2.8, label="5 drones"),
        Line2D([0], [0], color=DRONE_COLORS[6], linewidth=2.8, label="6 drones"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        title="Most efficient option",
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.7,
        labelspacing=0.45,
    )

    ax.set_xlabel("Payload mass - 10 [kg]")
    ax.set_ylabel(f"Range [km]")
    ax.set_xlim(0.0, PLOT_PAYLOAD_X_MAX_KG)
    ax.set_xticks(PLOT_PAYLOAD_TICKS_KG)
    ax.set_ylim(0.0, float(range_max))
    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def _plot_payload_efficiency(
    rows: list[dict[str, float | int | bool | str]],
    drone_counts: tuple[int, ...],
    output: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    curves: dict[int, dict[str, np.ndarray]] = {}

    for n_drones in drone_counts:
        group = [r for r in rows if int(r["n_drones"]) == n_drones]
        curves[n_drones] = {
            "payload": np.array([float(r["payload_kg"]) for r in group]),
            "feasible": np.array([bool(r["feasible"]) for r in group]),
            "wh": np.array([float(r["total_system_Wh_per_km"]) for r in group]),
            "draw_wh": np.array([
                float(r["unconstrained_total_system_Wh_per_km"]) for r in group
            ]),
        }

    def clipped_curve(n_drones: int) -> tuple[np.ndarray, np.ndarray]:
        curve = curves[n_drones]
        payload = curve["payload"] - PLOT_PAYLOAD_OFFSET_KG
        wh = curve["wh"]
        keep = (
            curve["feasible"]
            & (payload >= 0.0)
            & np.isfinite(wh)
        )
        next_counts = [n for n in drone_counts if n > n_drones]
        if next_counts:
            next_curve = curves[next_counts[0]]
            next_wh = next_curve["wh"]
            next_feasible = next_curve["feasible"] & np.isfinite(next_wh)
            crossing = np.flatnonzero(keep & next_feasible & (next_wh <= wh))
            if crossing.size:
                keep &= np.arange(keep.size) <= int(crossing[0])
        return payload[keep], wh[keep]

    plotted_wh: list[np.ndarray] = []
    for idx, n_drones in enumerate(drone_counts):
        line_zorder = 10 + len(drone_counts) - idx
        payload, wh_per_km = clipped_curve(n_drones)
        ax.plot(
            payload,
            wh_per_km,
            color=DRONE_COLORS[n_drones],
            linewidth=2.0,
            label=f"{n_drones} drones",
            zorder=line_zorder,
        )
        if payload.size:
            ax.scatter(
                payload[-1],
                wh_per_km[-1],
                color=DRONE_COLORS[n_drones],
                s=24,
                zorder=line_zorder,
            )
            plotted_wh.append(wh_per_km)

    ax.set_xlabel("Payload mass [kg]")
    ax.set_ylabel("Cruise fleet energy per distance [Wh/km]")
    ax.set_xlim(0.0, PLOT_PAYLOAD_X_MAX_KG)
    ax.set_xticks(PLOT_PAYLOAD_TICKS_KG)
    max_wh = max(float(np.nanmax(wh)) for wh in plotted_wh) if plotted_wh else 85.0
    ax.set_ylim(0.0, max(85.0, max_wh * 1.08))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Fleet size", loc="upper left")
    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def _payload_efficiency_rows(rows: list[dict[str, float | int | bool | str]]):
    return [
        {
            "n_drones": row["n_drones"],
            "payload_kg": row["payload_kg"],
            "payload_per_drone_kg": row["payload_per_drone_kg"],
            "fleet_Wh_per_km": row["total_system_Wh_per_km"],
            "fleet_climb_energy_Wh": row["total_system_climb_energy_Wh"],
            "battery_limited_range_km": row["battery_limited_range_km"],
            "CL": row["CL"],
            "CLmax_wing": row["CLmax_wing"],
            "CD_total": row["CD_total"],
            "thrust_required_per_drone_N": row["thrust_required_per_drone_N"],
            "thrust_available_per_drone_N": row["thrust_available_per_drone_N"],
            "feasible": row["feasible"],
            "reason": row["reason"],
        }
        for row in rows
    ]


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
                f"  {n_drones} drones: carries up to "
                f"{float(feasible[-1]['payload_kg']):.2f} kg, "
                f"battery range at that payload {float(feasible[-1]['battery_limited_range_km']):.1f} km"
            )
        if first_bad is not None:
            print(
                f"    cannot carry {float(first_bad['payload_kg']):.2f} kg: "
                f"{first_bad['reason']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("payload_range_efficiency.svg"))
    parser.add_argument("--csv", type=Path, default=Path("payload_range_efficiency.csv"))
    parser.add_argument("--option-csv", type=Path, default=Path("payload_efficiency_options.csv"))
    parser.add_argument("--payload-output", type=Path, default=Path("payload_efficiency.svg"))
    parser.add_argument("--payload-csv", type=Path, default=Path("payload_efficiency.csv"))
    parser.add_argument("--final-design", type=Path, default=Path("final_design_values.json"))
    parser.add_argument("--payload-max", type=float, default=120.0)
    parser.add_argument("--payload-step", type=float, default=0.25)
    parser.add_argument("--range-max", type=float, default=50.0)
    parser.add_argument("--range-step", type=float, default=0.25)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print(f"Loading final design from {args.final_design}...")
    final_design = _load_final_design(args.final_design)

    payloads = np.arange(0.0, args.payload_max + 0.5 * args.payload_step, args.payload_step)
    ranges_km = np.arange(0.0, args.range_max + 0.5 * args.range_step, args.range_step)
    raw_rows, drone_counts = _sweep(final_design, payloads)

    range_rows = [row.copy() for row in raw_rows]
    payload_values, grid, grid_rows = _best_grid(range_rows, ranges_km)
    _write_csv(range_rows, args.option_csv)
    _write_csv(grid_rows, args.csv)
    _plot(
        payload_values,
        ranges_km,
        grid,
        range_rows,
        drone_counts,
        args.payload_max,
        args.range_max,
        args.output,
        args.show,
    )

    energy_rows = [row.copy() for row in raw_rows]
    _write_csv(_payload_efficiency_rows(energy_rows), args.payload_csv)
    _plot_payload_efficiency(energy_rows, drone_counts, args.payload_output, args.show)

    per_drone_capacity_Wh = _battery_capacity_wh_from_mass(
        float(final_design["battery_mass_kg"]),
        n_cells=config.PROPULSION.n_cells,
        voltage_cell=config.PROPULSION.voltage_cell,
    )

    print("Final design loaded:")
    print(f"  mass per drone: {final_design['empty_mass_per_drone_kg']:.3f} kg")
    print(f"  battery mass per drone: {final_design['battery_mass_kg']:.3f} kg")
    print(
        "  usable battery check: "
        f"{per_drone_capacity_Wh * config.PROPULSION.DoD:.1f} Wh per drone "
        "multiplied by fleet size"
    )
    print(f"  wing span: {final_design['wing_span_m']:.3f} m")
    print(f"  aspect ratio: {final_design['aspect_ratio']:.3f}")
    print(f"  CLmax wing: {final_design['CLmax_wing']:.3f}")
    print(f"  max thrust per drone: {final_design['max_thrust_per_drone_N']:.1f} N")
    print("Payload and battery limits:")
    _print_limits(range_rows, drone_counts)
    print(f"Saved range plot: {args.output}")
    print(f"Saved selected-grid data: {args.csv}")
    print(f"Saved option data: {args.option_csv}")
    print(f"Saved payload-efficiency plot: {args.payload_output}")
    print(f"Saved payload-efficiency data: {args.payload_csv}")


if __name__ == "__main__":
    main()
