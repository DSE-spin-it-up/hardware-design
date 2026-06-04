"""Optimize tail-boom length for minimum mission energy.

This script varies ``sizing.L_boom`` while leaving the rest of ``config.yaml``
unchanged. Each candidate runs the full sizing pipeline on a throwaway config,
then reports the per-drone propulsion energy.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from pipeline import config, loop
from pipeline.sensitivity import clone_config


@dataclass
class BoomCandidate:
    L_boom: float
    energy_Wh: float
    mass_kg: float
    cd_full: float
    scissor_margin: float
    feasible: bool
    message: str = ""


def _scissor_margin(result) -> float:
    s = result.scissor
    required = float(np.interp(
        s.x_cg_current,
        s.x_cg,
        np.maximum(s.ShS_stab, s.ShS_ctrl),
    ))
    return float(s.ShS_current - required)


def evaluate_tail_boom(
    L_boom: float,
    *,
    quiet: bool = True,
    scissor_tol: float = 1.0e-3,
) -> BoomCandidate:
    sizing = dataclasses.replace(config.SIZING, L_boom=float(L_boom))
    cfg = clone_config(sizing=sizing, propulsion=config.PROPULSION)

    try:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = loop.run_pipeline(cfg)
        else:
            result = loop.run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 - optimization should survive bad points
        return BoomCandidate(
            L_boom=float(L_boom),
            energy_Wh=math.inf,
            mass_kg=math.inf,
            cd_full=math.inf,
            scissor_margin=-math.inf,
            feasible=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    margin = _scissor_margin(result)
    feasible = (
        result.lift_achievable
        and result.failure_lift_achievable
        and margin >= -scissor_tol
    )
    energy_Wh = result.propulsion.E_total / 3600.0
    if not feasible:
        energy_Wh = math.inf

    return BoomCandidate(
        L_boom=float(L_boom),
        energy_Wh=float(energy_Wh),
        mass_kg=float(result.masses["total"]),
        cd_full=float(result.cd_full_buildup),
        scissor_margin=margin,
        feasible=feasible,
        message="" if feasible else f"constraint infeasible (scissor_margin={margin:+.5f})",
    )


def _print_candidate(c: BoomCandidate) -> None:
    if c.feasible:
        print(
            f"  L_boom={c.L_boom:5.3f} m  "
            f"E={c.energy_Wh:8.2f} Wh  "
            f"m={c.mass_kg:6.3f} kg  "
            f"CD={c.cd_full:7.5f}  "
            f"scissor_margin={c.scissor_margin:+.5f}"
        )
    else:
        print(f"  L_boom={c.L_boom:5.3f} m  infeasible  {c.message}")


def sweep(
    lower: float,
    upper: float,
    samples: int,
    *,
    quiet: bool,
    scissor_tol: float,
) -> list[BoomCandidate]:
    candidates: list[BoomCandidate] = []
    for L_boom in np.linspace(lower, upper, samples):
        candidate = evaluate_tail_boom(
            float(L_boom),
            quiet=quiet,
            scissor_tol=scissor_tol,
        )
        _print_candidate(candidate)
        candidates.append(candidate)
    return candidates


def refine_around_best(
    best: BoomCandidate,
    *,
    lower: float,
    upper: float,
    bracket_width: float,
    maxiter: int,
    quiet: bool,
    scissor_tol: float,
) -> BoomCandidate:
    lo = max(lower, best.L_boom - bracket_width)
    hi = min(upper, best.L_boom + bracket_width)
    cache: dict[float, BoomCandidate] = {}

    def objective(x: float) -> float:
        key = round(float(x), 6)
        if key not in cache:
            cache[key] = evaluate_tail_boom(
                key,
                quiet=quiet,
                scissor_tol=scissor_tol,
            )
            _print_candidate(cache[key])
        return cache[key].energy_Wh

    opt = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-3, "maxiter": maxiter},
    )
    refined = cache.get(round(float(opt.x), 6))
    if refined is None:
        refined = evaluate_tail_boom(float(opt.x), quiet=quiet, scissor_tol=scissor_tol)
    return refined if refined.energy_Wh < best.energy_Wh else best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=float, default=0.5, help="Lower boom length bound [m]")
    parser.add_argument("--upper", type=float, default=3.0, help="Upper boom length bound [m]")
    parser.add_argument("--samples", type=int, default=9, help="Number of coarse sweep points")
    parser.add_argument("--refine", action="store_true", help="Run bounded refinement near the best sweep point")
    parser.add_argument("--refine-width", type=float, default=0.35, help="Half-width around best sweep point [m]")
    parser.add_argument("--maxiter", type=int, default=8, help="Maximum refinement evaluations")
    parser.add_argument("--scissor-tol", type=float, default=1.0e-3, help="Allowed negative scissor margin")
    parser.add_argument("--verbose", action="store_true", help="Show full pipeline output for each candidate")
    args = parser.parse_args()

    if args.upper > 3.0:
        raise ValueError("--upper must be <= 3.0 m")
    if args.lower <= 0.0 or args.lower >= args.upper:
        raise ValueError("Require 0 < --lower < --upper")
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")

    print("Tail-boom energy sweep")
    print(f"Bounds: {args.lower:.3f} m <= L_boom <= {args.upper:.3f} m")
    print(f"Baseline L_boom: {config.SIZING.L_boom:.3f} m")

    candidates = sweep(
        args.lower,
        args.upper,
        args.samples,
        quiet=not args.verbose,
        scissor_tol=args.scissor_tol,
    )
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        raise RuntimeError("No feasible tail-boom length found in the coarse sweep.")

    best = min(feasible, key=lambda c: c.energy_Wh)

    if args.refine:
        print("\nRefining near best sweep point")
        best = refine_around_best(
            best,
            lower=args.lower,
            upper=args.upper,
            bracket_width=args.refine_width,
            maxiter=args.maxiter,
            quiet=not args.verbose,
            scissor_tol=args.scissor_tol,
        )

    print("\nBest feasible design")
    _print_candidate(best)


if __name__ == "__main__":
    main()
