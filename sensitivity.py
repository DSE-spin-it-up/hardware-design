"""Sensitivity analysis entry point.

Edit the sweep lists below and run `python sensitivity.py`. For each design
variable the other knobs are held at their `config.yaml` baseline; the variable
is swept across its listed values, the full sizing pipeline is re-run at every
point, and the whole-fleet mass `m` and energy `E` (per-drone values ×
`n_drones`, payload excluded) are plotted on a twin-axis figure (mass left,
energy right) — one figure per variable. Sweeping `n_drones` therefore scales
the system totals linearly with the fleet size.

Pass a single sweep name to run it, e.g. `python sensitivity.py n_drones` or
`python sensitivity.py props`. Use `python sensitivity.py --all` to run every
sweep, or `python sensitivity.py --list` to see the available sweeps.
"""
from __future__ import annotations

import argparse

import numpy as np
from pathlib import Path

from pipeline import sensitivity

# ----- Sweep values -----
# Propeller table to iterate through (paths relative to the repo root).
CSV_PROPS = [
    # "data/10x8E_performance.csv",
    "data/11x8E_performance.csv",
    "data/12x8E_performance.csv",
    "data/13x8E_performance.csv",
    "data/14x8_performance.csv",
    # "data/15x8E_performance.csv",
]
N_DRONES = [2, 3, 4, 5, 6]                              # [-]
V_CRUISE = [15, 17.5, 20, 22.5, 25]                         # [m/s]
RANGE_M = [10_000, 15_000, 20_000, 25_000, 30_000]     # [m]


def _prop_label(path: str) -> str:
    """`data/14x10E_performance.csv` -> `14x10E`."""
    return Path(path).stem.replace("_performance", "")


# ----- Sweep registry -----
# Keyed by the command-line name. ``variable`` is the ``evaluate`` keyword that
# is actually swept; ``values`` are the sweep points; the rest are plot labels.
# ``x_transform`` (optional) maps the raw sweep values to the x values shown on
# the plot; ``categorical`` flags non-numeric x axes.
SWEEPS: dict[str, dict] = {
    "n_drones": dict(
        variable="n_drones",
        values=N_DRONES,
        xlabel="Number of drones  [-]",
        title="Sensitivity to number of drones",
        integer_x=True,
    ),
    "V_cruise": dict(
        variable="V_cruise",
        values=V_CRUISE,
        xlabel="Cruise velocity  [m/s]",
        title="Sensitivity to cruise velocity",
    ),
    "Range": dict(
        variable="R",
        values=RANGE_M,
        xlabel="Range  [km]",
        title="Sensitivity to range",
        x_transform=lambda xs: np.array(xs) / 1000.0,
    ),
    "props": dict(
        variable="csv_prop",
        values=CSV_PROPS,
        xlabel="Propeller",
        title="Sensitivity to propeller",
        categorical=True,
        x_transform=lambda xs: [_prop_label(p) for p in xs],
    ),
}


def run_sweep(name: str, show_plots: bool = True) -> None:
    """Run a single registered sweep and render its twin-axis figure."""
    spec = SWEEPS[name]
    print(f">>> Sensitivity: {spec['title']}")
    xs, m, e = sensitivity.sweep_one(spec["variable"], spec["values"])
    x_plot = spec.get("x_transform", lambda v: v)(xs)
    sensitivity.plot_dual_axis(
        x_plot, m, e,
        xlabel=spec["xlabel"], title=spec["title"],
        categorical=spec.get("categorical", False),
        integer_x=spec.get("integer_x", False), show=show_plots,
    )


def main(sweep: str | None = None, show_plots: bool = True) -> None:
    """Run a single sweep, or all of them when ``sweep`` is None."""
    for name in ([sweep] if sweep else list(SWEEPS)):
        run_sweep(name, show_plots=show_plots)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-factor-at-a-time sensitivity sweeps.",
    )
    parser.add_argument(
        "sweep", nargs="?", choices=list(SWEEPS), metavar="SWEEP",
        help="sweep to run; omit to run all "
             f"({', '.join(SWEEPS)})",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the available sweeps and exit",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="run all sweeps; this may take a long time",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="compute the sweep without opening plot windows",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.list:
        for name, spec in SWEEPS.items():
            print(f"{name:<10} {spec['title']}")
    else:
        if args.sweep is None and not args.all:
            print("Choose a sweep to run, or pass --all for the full batch.")
            print("Available sweeps:")
            for name, spec in SWEEPS.items():
                print(f"  {name:<10} {spec['title']}")
        else:
            main(sweep=None if args.all else args.sweep, show_plots=not args.no_show)
