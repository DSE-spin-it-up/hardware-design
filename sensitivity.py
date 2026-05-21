"""Sensitivity analysis entry point.

Edit the sweep lists below and run `python sensitivity.py`. For each design
variable the other knobs are held at their `config.yaml` baseline; the variable
is swept across its listed values, the full sizing pipeline is re-run at every
point, and per-drone total mass `m` and total energy `E` are plotted on a
twin-axis figure (mass left, energy right) — one figure per variable.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from pipeline import sensitivity

# ----- Sweep values -----
# Propeller table to iterate through (paths relative to the repo root).
CSV_PROPS = [
    "data/14x10E_performance.csv",
    "data/15x10E_performance.csv",
    "data/13x10E_performance.csv",
    "data/12x10W_performance.csv",
    "data/20x10E_performance.csv",
]
N_DRONES = [2, 3, 4, 5, 6]                              # [-]
V_CRUISE = [15, 17.5, 20, 22.5, 25]                         # [m/s]
RANGE_M = [10_000, 15_000, 20_000, 25_000, 30_000]     # [m]


def _prop_label(path: str) -> str:
    """`data/14x10E_performance.csv` -> `14x10E`."""
    return Path(path).stem.replace("_performance", "")


def main(show_plots: bool = True) -> None:
    print(">>> Sensitivity: number of drones")
    xs, m, e = sensitivity.sweep_one("n_drones", N_DRONES)
    sensitivity.plot_dual_axis(
        xs, m, e, xlabel="number of drones  [-]",
        title="Sensitivity to number of drones", show=show_plots,
    )

    print(">>> Sensitivity: cruise velocity")
    xs, m, e = sensitivity.sweep_one("V_cruise", V_CRUISE)
    sensitivity.plot_dual_axis(
        xs, m, e, xlabel="cruise velocity  [m/s]",
        title="Sensitivity to cruise velocity", show=show_plots,
    )

    print(">>> Sensitivity: range")
    xs, m, e = sensitivity.sweep_one("R", RANGE_M)
    sensitivity.plot_dual_axis(
        np.array(xs) / 1000.0, m, e, xlabel="range  [km]",
        title="Sensitivity to range", show=show_plots,
    )

    print(">>> Sensitivity: propeller")
    xs, m, e = sensitivity.sweep_one("csv_prop", CSV_PROPS)
    sensitivity.plot_dual_axis(
        [_prop_label(p) for p in xs], m, e, xlabel="propeller",
        title="Sensitivity to propeller", categorical=True, show=show_plots,
    )


if __name__ == "__main__":
    main()
