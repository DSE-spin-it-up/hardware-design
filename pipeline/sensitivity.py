"""One-factor-at-a-time sensitivity analysis for system mass and energy.

For each design variable, hold the other knobs at their ``config.yaml`` baseline,
sweep the chosen variable across a list of values, re-run the full sizing
pipeline at every point, and read off the *whole-fleet* mass ``m`` and energy
``E`` (per-drone values × ``n_drones``, payload excluded). Each sweep is rendered
as a twin-axis plot: mass on the left y-axis, energy on the right.

The driver script lives at the repo root (``sensitivity.py``); this module holds
the reusable machinery. Nothing here mutates ``pipeline.config`` — every run
gets a throwaway clone of the config with a couple of fields overridden.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline import config as base_config
from pipeline import loop


def _resolve_csv(value: str) -> str:
    """Anchor a relative propeller-CSV path against the repo root.

    Mirrors ``pipeline.config._resolve_repo_path`` so a sweep can name props as
    plain ``data/<file>.csv`` regardless of the cwd it is launched from.
    """
    p = Path(value)
    if p.is_absolute():
        return value
    return str((base_config.PROJECT_ROOT / p).resolve())


def clone_config(*, sizing, propulsion) -> types.SimpleNamespace:
    """Throwaway copy of ``pipeline.config`` with SIZING/PROPULSION swapped.

    ``run_pipeline`` only reads the module-level UPPERCASE constants, so we copy
    all of them onto a ``SimpleNamespace`` and override the two dataclasses we
    vary. The original module is left untouched.
    """
    attrs = {name: getattr(base_config, name)
             for name in dir(base_config) if name.isupper()}
    attrs["SIZING"] = sizing
    attrs["PROPULSION"] = propulsion
    return types.SimpleNamespace(**attrs)


def evaluate(
    *,
    n_drones: int | None = None,
    V_cruise: float | None = None,
    R: float | None = None,
    csv_prop: str | None = None,
) -> tuple[float, float, float, float]:
    """Run the pipeline for one design point.

    Returns ``(m_system, E_system, m_per_drone, E_per_drone)`` with mass in kg
    and energy in Wh. The ``_system`` values are *whole-fleet* totals: the
    per-drone empty mass and per-drone cruise+climb energy multiplied by
    ``n_drones`` (payload excluded), so sweeping ``n_drones`` scales the fleet
    linearly. The ``_per_drone`` values are the single-drone figures.

    Any argument left at ``None`` keeps its ``config.yaml`` baseline. Solver
    failures at extreme sweep values (prop non-convergence, motor KV out of fit
    range, …) are caught and returned as all-``nan`` so one bad point does
    not abort the whole sweep — it just shows up as a gap in the plot.
    """
    sizing_over: dict = {}
    if n_drones is not None:
        sizing_over["n_drones"] = n_drones
    if V_cruise is not None:
        sizing_over["V_cruise"] = V_cruise
    if R is not None:
        sizing_over["R"] = R
    sizing = (dataclasses.replace(base_config.SIZING, **sizing_over)
              if sizing_over else base_config.SIZING)

    propulsion = base_config.PROPULSION
    if csv_prop is not None:
        propulsion = dataclasses.replace(propulsion, csv_prop=_resolve_csv(csv_prop))

    cfg = clone_config(sizing=sizing, propulsion=propulsion)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = loop.run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 — any solver failure → NaN point
        print(f"    ! run failed: {type(exc).__name__}: {exc}")
        nan = float("nan")
        return nan, nan, nan, nan
    # Per-drone figures, then whole-fleet totals: scale the per-drone empty mass
    # and per-drone energy by the number of drones (payload excluded —
    # hardware-only system mass).
    n = result.sizing.inputs.n_drones
    m_per_drone = result.masses["total"]
    E_per_drone = result.propulsion.E_total / 3600.0
    m_system = m_per_drone * n
    E_system = E_per_drone * n
    return m_system, E_system, m_per_drone, E_per_drone


def sweep_one(
    variable: str, values
) -> tuple[list, np.ndarray, np.ndarray]:
    """OAT sweep of a single ``evaluate`` keyword across ``values``.

    Returns ``(values, masses [kg], energies [Wh])`` with NaNs for any point
    whose pipeline run failed.
    """
    masses: list[float] = []
    energies: list[float] = []
    for v in values:
        m, e, m_pd, e_pd = evaluate(**{variable: v})
        print(f"    {variable} = {v!r:<28} "
              f"fleet: m = {m:7.3f} kg  E = {e:8.2f} Wh   "
              f"per drone: m = {m_pd:6.3f} kg  E = {e_pd:7.2f} Wh")
        masses.append(m)
        energies.append(e)
    return list(values), np.array(masses), np.array(energies)


def plot_dual_axis(
    xs,
    masses: np.ndarray,
    energies: np.ndarray,
    *,
    xlabel: str,
    title: str,
    categorical: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Twin-axis plot: fleet mass on the left y-axis, fleet energy on the right.

    Set ``categorical=True`` for non-numeric x values (e.g. propeller names),
    which are placed at evenly spaced ticks and labelled by ``xs``.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x_plot = np.arange(len(xs)) if categorical else np.asarray(xs, dtype=float)
    (l_m,) = ax1.plot(x_plot, masses, "o-", color="C0", label="system mass $m$")
    (l_e,) = ax2.plot(x_plot, energies, "s--", color="C3", label="system energy $E$")

    if categorical:
        ax1.set_xticks(x_plot)
        ax1.set_xticklabels([str(x) for x in xs], rotation=30, ha="right")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("system mass $m$ (fleet)  [kg]", color="C0")
    ax2.set_ylabel("system energy $E$ (fleet)  [Wh]", color="C3")
    ax1.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C3")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=[l_m, l_e], loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
