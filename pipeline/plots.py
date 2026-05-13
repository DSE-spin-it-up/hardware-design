"""Pipeline plots: convergence history and drone L/D polars."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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
