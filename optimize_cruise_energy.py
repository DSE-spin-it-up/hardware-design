"""Grid-search battery height and tail-boom length for minimum cruise energy.

This is intentionally separate from ``main.py``. Normal design runs still use
the pipeline's built-in battery-y placement. This script opts into a 2D sweep:

  - ``sizing.L_boom`` is varied in meters.
  - ``battery_y_frac`` is varied from 0.0 (lowest feasible battery bottom) to
    1.0 (highest feasible battery bottom) for each candidate geometry.

Each grid point runs the full sizing pipeline on a throwaway config object.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import io
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from pipeline import config, loop
from pipeline.sensitivity import clone_config


@dataclass
class EnergyCandidate:
    L_boom: float
    battery_y_frac: float
    feasible: bool
    cruise_energy_Wh: float = math.inf
    total_energy_Wh: float = math.inf
    cruise_power_W: float = math.inf
    mass_kg: float = math.inf
    cd_full: float = math.inf
    ShS: float = math.inf
    Sh: float = math.inf
    battery_y0: float = math.nan
    battery_y: float = math.nan
    x_cg: float = math.nan
    y_cg: float = math.nan
    scissor_margin: float = -math.inf
    tail_cl_margin: float = -math.inf
    message: str = ""


def _scissor_margin(result) -> float:
    s = result.scissor
    required = float(np.interp(
        s.x_cg_current,
        s.x_cg,
        np.maximum(s.ShS_stab, s.ShS_ctrl),
    ))
    return float(s.ShS_current - required)


def _tail_cl_margin(result) -> float:
    return float(result.elevator.tail_CL_limit - max(
        abs(result.elevator.CLh_at_max_up),
        abs(result.elevator.CLh_at_max_down),
    ))


def _candidate_config(*, L_boom: float, battery_y_frac: float):
    sizing = dataclasses.replace(config.SIZING, L_boom=float(L_boom))
    cfg = clone_config(sizing=sizing, propulsion=config.PROPULSION)
    cfg.BATTERY_Y0 = None
    cfg.BATTERY_Y_FRAC = float(battery_y_frac)
    return cfg


def evaluate(
    L_boom: float,
    battery_y_frac: float,
    *,
    objective: str,
    quiet: bool,
    scissor_tol: float,
    tail_margin_tol: float,
) -> EnergyCandidate:
    cfg = _candidate_config(L_boom=L_boom, battery_y_frac=battery_y_frac)
    try:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = loop.run_pipeline(cfg)
        else:
            result = loop.run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 - a bad grid point should not abort the sweep
        return EnergyCandidate(
            L_boom=float(L_boom),
            battery_y_frac=float(battery_y_frac),
            feasible=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    scissor_margin = _scissor_margin(result)
    tail_cl_margin = _tail_cl_margin(result)
    feasible = (
        result.lift_achievable
        and result.failure_lift_achievable
        and scissor_margin >= -scissor_tol
        and tail_cl_margin >= -tail_margin_tol
    )
    cruise_energy_Wh = float(result.propulsion.E_cruise / 3600.0)
    total_energy_Wh = float(result.propulsion.E_total / 3600.0)
    if not feasible:
        if objective == "cruise":
            cruise_energy_Wh = math.inf
        else:
            total_energy_Wh = math.inf

    return EnergyCandidate(
        L_boom=float(L_boom),
        battery_y_frac=float(battery_y_frac),
        feasible=feasible,
        cruise_energy_Wh=cruise_energy_Wh,
        total_energy_Wh=total_energy_Wh,
        cruise_power_W=float(result.propulsion.P_elec_cruise),
        mass_kg=float(result.masses["total"]),
        cd_full=float(result.cd_full_buildup),
        ShS=float(result.sizing.Sh / result.sizing.Sw),
        Sh=float(result.sizing.Sh),
        battery_y0=float(result.battery_y0),
        battery_y=float(result.y_cg["battery"]),
        x_cg=float(result.cg["overall"]),
        y_cg=float(result.y_cg["overall"]),
        scissor_margin=scissor_margin,
        tail_cl_margin=tail_cl_margin,
        message="" if feasible else (
            f"constraint infeasible "
            f"(scissor={scissor_margin:+.5f}, tailCL={tail_cl_margin:+.5f})"
        ),
    )


def _score(candidate: EnergyCandidate, objective: str) -> float:
    return candidate.cruise_energy_Wh if objective == "cruise" else candidate.total_energy_Wh


def _print_candidate(candidate: EnergyCandidate, objective: str) -> None:
    if not candidate.feasible:
        print(
            f"  L={candidate.L_boom:5.3f} m  "
            f"yfrac={candidate.battery_y_frac:4.2f}  infeasible  {candidate.message}"
        )
        return
    score = _score(candidate, objective)
    label = "Ecruise" if objective == "cruise" else "Etotal"
    print(
        f"  L={candidate.L_boom:5.3f} m  "
        f"yfrac={candidate.battery_y_frac:4.2f}  "
        f"{label}={score:8.2f} Wh  "
        f"Pcruise={candidate.cruise_power_W:7.1f} W  "
        f"m={candidate.mass_kg:6.3f} kg  "
        f"Sh/S={candidate.ShS:6.4f}  "
        f"tailCL={candidate.tail_cl_margin:+.4f}"
    )


def _linspace(lower: float, upper: float, samples: int) -> np.ndarray:
    if samples < 2:
        return np.array([lower], dtype=float)
    return np.linspace(lower, upper, samples)


def _write_csv(path: Path, candidates: list[EnergyCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(EnergyCandidate)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(dataclasses.asdict(candidate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boom-lower", type=float, default=1.0, help="Lower L_boom bound [m]")
    parser.add_argument("--boom-upper", type=float, default=3.0, help="Upper L_boom bound [m]")
    parser.add_argument("--boom-samples", type=int, default=5, help="Number of L_boom samples")
    parser.add_argument("--battery-samples", type=int, default=5, help="Number of battery-y fraction samples")
    parser.add_argument("--objective", choices=("cruise", "total"), default="cruise",
                        help="Energy objective to rank candidates")
    parser.add_argument("--scissor-tol", type=float, default=1.0e-3,
                        help="Allowed negative scissor margin")
    parser.add_argument("--tail-margin-tol", type=float, default=1.0e-4,
                        help="Allowed negative elevator tail-CL margin")
    parser.add_argument("--csv", type=Path, default=Path("cruise_energy_sweep.csv"),
                        help="CSV output path")
    parser.add_argument("--verbose", action="store_true", help="Show full pipeline output per candidate")
    args = parser.parse_args()

    if args.boom_upper > 3.0:
        raise ValueError("--boom-upper must be <= 3.0 m")
    if args.boom_lower <= 0.0 or args.boom_lower > args.boom_upper:
        raise ValueError("Require 0 < --boom-lower <= --boom-upper")
    if args.boom_samples < 1 or args.battery_samples < 1:
        raise ValueError("Sample counts must be at least 1")

    boom_values = _linspace(args.boom_lower, args.boom_upper, args.boom_samples)
    battery_fracs = _linspace(0.0, 1.0, args.battery_samples)

    print("Cruise-energy 2D sweep")
    print(f"L_boom: {boom_values[0]:.3f} to {boom_values[-1]:.3f} m ({len(boom_values)} samples)")
    print(f"battery_y_frac: 0.00 to 1.00 ({len(battery_fracs)} samples)")
    print(f"objective: {args.objective}")

    candidates: list[EnergyCandidate] = []
    for L_boom in boom_values:
        for battery_y_frac in battery_fracs:
            candidate = evaluate(
                float(L_boom),
                float(battery_y_frac),
                objective=args.objective,
                quiet=not args.verbose,
                scissor_tol=args.scissor_tol,
                tail_margin_tol=args.tail_margin_tol,
            )
            candidates.append(candidate)
            _print_candidate(candidate, args.objective)

    _write_csv(args.csv, candidates)

    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        print(f"\nWrote {args.csv}, but no feasible grid point was found.")
        return

    ranked = sorted(feasible, key=lambda candidate: _score(candidate, args.objective))
    best = ranked[0]

    print(f"\nWrote {args.csv}")
    print("Best feasible design")
    _print_candidate(best, args.objective)
    print(
        f"    battery_y0={best.battery_y0:.4f} m, "
        f"battery_y={best.battery_y:.4f} m, "
        f"x_cg={best.x_cg:.4f} m, y_cg={best.y_cg:.4f} m, "
        f"Sh={best.Sh:.4f} m^2"
    )


if __name__ == "__main__":
    main()
