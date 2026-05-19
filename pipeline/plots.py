"""Pipeline plots: convergence history, drone L/D polars, and CG side view."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

from aerodynamics.airfoil_geometry import AirfoilGeometry
from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from structures.rods import RodResult
from weights.mass import compute_y_cg


def plot_convergence(
    CD0_history: list[float],
    mass_history: list[float],
    Sw_history: list[float],
    cg_history: list[float],
    Cd0_guess: float,
    m_drone_guess: float,
    Sw_guess: float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    cd_iters = np.arange(len(CD0_history) + 1)
    cd_values = np.array([Cd0_guess, *CD0_history])
    axes[0].plot(cd_iters, cd_values, "o-")
    axes[0].axhline(cd_values[-1], color="g", ls="--",
                    label=f"converged CD0 = {cd_values[-1]:.5f}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("CD0")
    axes[0].set_title("CD0 progression")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    m_iters = np.arange(len(mass_history) + 1)
    m_values = np.array([m_drone_guess, *mass_history])
    axes[1].plot(m_iters, m_values, "o-")
    axes[1].axhline(m_values[-1], color="g", ls="--",
                    label=f"converged m_drone = {m_values[-1]:.3f} kg")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("drone mass [kg]")
    axes[1].set_title("Drone mass progression")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    sw_iters = np.arange(len(Sw_history) + 1)
    sw_values = np.array([Sw_guess, *Sw_history])
    axes[2].plot(sw_iters, sw_values, "o-")
    axes[2].axhline(sw_values[-1], color="g", ls="--",
                    label=f"converged Sw = {sw_values[-1]:.4f} m²")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("wing area Sw [m²]")
    axes[2].set_title("Wing area progression")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # CG only exists from iter 1 onward (depends on propulsion result).
    cg_iters = np.arange(1, len(cg_history) + 1)
    cg_values = np.array(cg_history)
    axes[3].plot(cg_iters, cg_values, "o-")
    axes[3].axhline(cg_values[-1], color="g", ls="--",
                    label=f"converged x_cg = {cg_values[-1]:.4f} m")
    cg_excursion = cg_values.max() - cg_values.min()
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("x_cg from LEMAC [m]")
    axes[3].set_title(f"CG excursion (Δ = {cg_excursion * 1000:.1f} mm)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    plt.show()


def plot_drone_ld(
    CL: np.ndarray,
    CD_drone: np.ndarray,
    CD_full: np.ndarray,
    CL_op: float,
    airfoil_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, CD, title in [
        (axes[0], CD_drone, "Drone only"),
        (axes[1], CD_full, "Drone + payload"),
    ]:
        LD = CL / CD
        idx_max = int(np.argmax(LD))
        ax.plot(CL, LD, "o-", label="L/D")
        ax.axvline(CL_op, color="r", ls="--", label=f"operating CL = {CL_op:.3f}")
        ax.plot(
            CL[idx_max], LD[idx_max], "g*", ms=14,
            label=f"max L/D = {LD[idx_max]:.1f} @ CL = {CL[idx_max]:.2f}",
        )
        ax.set_xlabel("CL")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("L/D")
    fig.suptitle(f"Drag polar — airfoil {airfoil_name}")
    fig.tight_layout()
    plt.show()


def plot_cg_side_view(
    sizing: SizingResult,
    fus: FuselageResult,
    struct: RodResult,
    cg: dict[str, float],
    masses: dict[str, float],
    airfoil_path: str | Path,
    show: bool = True,
) -> plt.Figure:
    """Side-view sketch of the aircraft with component CGs and overall CG.

    Draws the fuselage box, wing airfoil polygon, battery, PVC tube + rods,
    tail rod, and scatter markers for motors / tail / overall CG. The
    overall vertical CG is computed from each component's vertical position
    weighted by its mass.
    """
    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    airfoil_coords = np.column_stack((
        airfoil.polygon[:, 0] * root_chord,
        airfoil.polygon[:, 1] * root_chord,
    ))
    airfoil_coords[:, 1] -= float(np.min(airfoil_coords[:, 1]))
    airfoil_height = float(np.max(airfoil_coords[:, 1]))

    x_motor = cg["motors"]
    x_rod_wing = cg["rod_spar"]
    x_rod_aileron = cg["rod_aileron"]
    x_pvc = cg["pvc_tubes"]
    x_batt = cg["battery"]
    x_tail = cg["tail"]
    x_overall = cg["overall"]
    # Tail LE position from LEMAC: tail AC sits at 0.25·c + lh; LE is 0.25·ct ahead.
    x_tail_root = 0.25 * sizing.c + sizing.lh - 0.25 * sizing.ct

    rod_radius_w = struct.d_w / 2.0
    rod_radius_a = struct.d_aileron / 2.0
    battery_length = fus.battery_length
    battery_height = fus.battery_height
    rod_span_length = abs(x_rod_aileron - x_rod_wing)
    tube_height = max(max(struct.d_spar, struct.d_aileron) * 1.1, 0.03)
    tube_length = max(rod_span_length + 0.1, battery_length * 0.8)
    tube_x0 = max(x_pvc - tube_length / 2.0, 0.0)
    tube_y0 = max(airfoil_height - tube_height - 0.005, 0.0)
    batt_x0 = x_batt - battery_length / 2.0
    batt_y0 = tube_y0 + tube_height + 0.005
    plot_height = max(fus.height, batt_y0 + battery_height + 0.01)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.add_patch(Rectangle((0.0, 0.0), fus.length, fus.height,
                           fill=False, linewidth=2, label="Fuselage"))
    ax.add_patch(Polygon(airfoil_coords, closed=True,
                         facecolor="lightblue", edgecolor="navy", alpha=0.6,
                         label="Wing profile"))
    ax.add_patch(Rectangle((batt_x0, batt_y0), battery_length, battery_height,
                           color="orange", alpha=0.5, label="Battery"))
    ax.add_patch(Rectangle((tube_x0, tube_y0), tube_length, tube_height,
                           facecolor="lightgreen", alpha=0.4, edgecolor="darkgreen",
                           label="PVC tube"))
    ax.add_patch(Ellipse((tube_x0, tube_y0 + tube_height / 2.0),
                         tube_height, tube_height,
                         facecolor="lightgreen", edgecolor="darkgreen", alpha=0.4))
    ax.add_patch(Ellipse((tube_x0 + tube_length, tube_y0 + tube_height / 2.0),
                         tube_height, tube_height,
                         facecolor="lightgreen", edgecolor="darkgreen", alpha=0.4))

    rod_y = tube_y0 + tube_height / 2.0
    ax.add_patch(Circle((x_rod_wing, rod_y), rod_radius_w,
                        color="brown", alpha=0.8, label="Wing rod"))
    ax.add_patch(Circle((x_rod_aileron, rod_y), rod_radius_a,
                        color="sienna", alpha=0.8, label="Aileron rod"))

    tail_start = x_rod_aileron + rod_radius_a + 0.001
    ax.plot([tail_start, x_tail_root], [rod_y, rod_y],
            color="gray", linewidth=3, solid_capstyle="butt", label="Tail rod")

    y_cg = compute_y_cg(sizing, fus, struct, masses, airfoil_path)
    motor_y = y_cg["motors"]
    tail_y = y_cg["tail"]
    overall_y = y_cg["overall"]
    plot_height = max(plot_height, tail_y + 0.05)

    ax.scatter([x_motor, x_tail, x_overall], [motor_y, tail_y, overall_y],
               color=["red", "purple", "black"], zorder=5)
    ax.text(x_motor, motor_y + 0.03, "Motors", color="red", ha="center")
    ax.text(x_tail, tail_y + 0.02, "Tail", color="purple", ha="center")
    ax.text(x_overall, overall_y + 0.02, "Overall CG", color="black", ha="center")

    ax.set_title("Aircraft CG side view")
    ax.set_xlabel("x [m] from LEMAC")
    ax.set_ylabel("vertical position [m]")
    ax.set_xlim(-0.05, max(fus.length, x_tail, x_overall) + 0.2)
    ax.set_ylim(-0.05, plot_height + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    if show:
        plt.show()
    return fig


def plot_scissor(scissor, *, show: bool = True):
    """Scissor plot: required S_h/S vs x_cg from LEMAC (stability + controllability).

    Green region above BOTH lines is the feasible design space — for any x_cg,
    the actual S_h/S must satisfy both constraints. The orange ⭕ marks the
    current operating point (converged x_cg, current S_h/S).
    """
    s = scissor
    fig, ax = plt.subplots(figsize=(9, 5))

    stab_pos = np.clip(s.ShS_stab, 0.0, None)
    ctrl_pos = np.clip(s.ShS_ctrl, 0.0, None)
    envelope = np.maximum(stab_pos, ctrl_pos)
    y_top = max(
        s.ShS_current * 1.6,
        float(np.nanmax(envelope)) * 1.15,
        0.05,
    )

    ax.plot(s.x_cg, s.ShS_stab, color="C0", lw=2.0, label="Stability boundary")
    ax.plot(s.x_cg, s.ShS_ctrl, color="C3", lw=2.0, label="Controllability boundary")
    ax.fill_between(s.x_cg, envelope, y_top,
                    color="green", alpha=0.12, label="Feasible")
    ax.fill_between(s.x_cg, 0.0, envelope,
                    color="red", alpha=0.08, label="Infeasible")

    ax.axhline(s.ShS_current, color="k", ls="--", lw=1.2,
               label=f"current $S_h/S$ = {s.ShS_current:.3f}")
    ax.axvline(s.x_cg_current, color="orange", ls="--", lw=1.2,
               label=f"current $x_{{cg}}$ = {s.x_cg_current:.4f} m "
                     f"({s.x_cg_current / s.c:.2%} $\\bar c$)")
    ax.plot([s.x_cg_current], [s.ShS_current], "o", color="orange",
            mec="k", ms=10, zorder=5)

    # Twin top axis showing x_cg as a fraction of MAC.
    ax2 = ax.twiny()
    ax2.set_xlim(s.x_cg[0] / s.c, s.x_cg[-1] / s.c)
    ax2.set_xlabel(r"$x_{cg} / \bar{c}$  [-]")

    ax.set_xlim(s.x_cg[0], s.x_cg[-1])
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel("$x_{cg}$ from LEMAC  [m]")
    ax.set_ylabel("$S_h / S$  [-]")
    ax.set_title(
        f"Scissor — SM={s.SM:.2f}, "
        f"$d\\varepsilon/d\\alpha$={s.dep_da:.3f}, "
        f"$V_h/V$={s.Vh_V:.2f},  "
        f"$C_{{L_h}}$={s.CL_h:+.3f},  $C_{{L_{{A-h}}}}$={s.CL_A_h:.3f},  "
        f"$C_{{m_{{ac}}}}$={s.Cm_ac:+.4f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
