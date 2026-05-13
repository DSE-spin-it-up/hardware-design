"""
llt.py — Prandtl Lifting-Line Theory (classical Fourier-series form).

Given a wing geometry, a 2-D airfoil polar, and a flight condition, solve the
monoplane equation for the Fourier coefficients A_n and derive:

    CL          — wing lift coefficient
    CD_i        — induced drag coefficient
    e           — Oswald span efficiency (1 / (1 + δ))
    Gamma(y)    — spanwise circulation distribution [m^2/s]
    ell(y)      — spanwise lift distribution (per unit span)  [N/m]

Math (matches llt_nvm_spec.md §2.2). With y = -(b/2)·cos(θ), θ ∈ [0, π]:

    Γ(θ) = 2·b·V∞ · Σ_{n=1..N} A_n · sin(n·θ)

Monoplane equation at each collocation point θ_k:

    μ(θ_k)·[α(θ_k) - α_L0(θ_k)]·sin(θ_k)
        = Σ_n A_n · sin(n·θ_k) · [sin(θ_k) + n·μ(θ_k)]

with μ(θ) = c(θ)·Cl_α / (4·b). Then:

    CL    = π · AR · A_1
    δ     = Σ_{n>=2} n · (A_n / A_1)^2
    CD_i  = CL^2 / (π · AR) · (1 + δ)
    e     = 1 / (1 + δ)

Scope: unswept, untwisted (or arbitrarily-twisted), variable-taper wings with
moderate to high AR. Sweep and dihedral are out of scope for this version (see
llt_nvm_spec.md §7).

Sign conventions (matches llt_nvm_spec.md §4):
    y spanwise, +starboard, 0 at root, ±b/2 at tip
    α positive nose-up, Γ positive for lift in +z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aerodynamics.airfoil_polar import AirfoilPolar


# ---------------------------------------------------------------------------
# Geometry and flight condition
# ---------------------------------------------------------------------------


@dataclass
class WingGeometry:
    """Planform definition.

    Provide any two of (b, S, AR); the third is derived. Taper ratio
    λ = c_tip / c_root is free; sweep and twist default to zero.

    A custom chord distribution can be supplied by passing `chord_func`
    (overrides taper-based chord). A custom twist by passing `twist_func`.
    `alpha_L0_func` is filled in from the airfoil polar at solve-time if
    left as None.
    """

    b: float | None = None
    S: float | None = None
    AR: float | None = None
    taper: float = 1.0  # λ = c_tip / c_root
    sweep_le: float = 0.0  # [rad] leading-edge sweep — informational, must stay 0
    chord_func: Callable[[np.ndarray], np.ndarray] | None = None
    twist_func: Callable[[np.ndarray], np.ndarray] | None = None
    alpha_L0_func: Callable[[np.ndarray], np.ndarray] | None = None

    # Filled in by __post_init__:
    c_root: float = field(init=False, default=0.0)
    c_tip: float = field(init=False, default=0.0)
    c_mean: float = field(init=False, default=0.0)

    def __post_init__(self):
        b, S, AR = self.b, self.S, self.AR
        n_given = sum(x is not None for x in (b, S, AR))
        if n_given < 2:
            raise ValueError("WingGeometry: supply at least two of (b, S, AR).")
        if b is None:
            self.b = float(np.sqrt(AR * S))
        elif S is None:
            self.S = float(self.b**2 / AR)
        elif AR is None:
            self.AR = float(self.b**2 / S)
        else:
            # All three given — verify consistency.
            if not np.isclose(self.b**2 / self.S, self.AR, rtol=1e-6):
                raise ValueError(
                    f"WingGeometry: inconsistent (b={b}, S={S}, AR={AR}); "
                    f"b^2/S = {self.b**2 / self.S:.6f} != AR."
                )

        if not (0.0 < self.taper <= 1.0):
            raise ValueError(f"taper must lie in (0, 1], got {self.taper}.")
        if not np.isclose(self.sweep_le, 0.0):
            raise NotImplementedError(
                "sweep_le must be 0 in this version (classical LLT)."
            )

        # Reference chords for a linear-taper planform: S = b · c_root · (1 + λ)/2.
        self.c_root = 2.0 * self.S / (self.b * (1.0 + self.taper))
        self.c_tip = self.c_root * self.taper
        self.c_mean = self.S / self.b

    # ----- callables on y -------------------------------------------------

    def chord(self, y: np.ndarray) -> np.ndarray:
        """Local chord c(y) [m]. Defaults to linear taper, symmetric about y=0."""
        if self.chord_func is not None:
            return np.asarray(self.chord_func(y), dtype=float)
        eta = np.abs(2.0 * np.asarray(y, dtype=float) / self.b)
        return self.c_root * (1.0 - (1.0 - self.taper) * eta)

    def twist(self, y: np.ndarray) -> np.ndarray:
        """Geometric twist θ(y) [rad]. Zero by default."""
        if self.twist_func is not None:
            return np.asarray(self.twist_func(y), dtype=float)
        return np.zeros_like(np.asarray(y, dtype=float))

    def alpha_L0(self, y: np.ndarray, polar_alpha_L0: float) -> np.ndarray:
        """Section zero-lift angle [rad]. Constant from the airfoil polar by default."""
        if self.alpha_L0_func is not None:
            return np.asarray(self.alpha_L0_func(y), dtype=float)
        return np.full_like(np.asarray(y, dtype=float), polar_alpha_L0)


@dataclass
class FlightCondition:
    """Free-stream and trim target.

    Provide exactly one of `alpha_root` (rad, geometric AoA at the root) or
    `CL_target` (dimensionless). When `CL_target` is set, the solver picks
    `alpha_root` such that the wing CL matches.
    """

    V_inf: float
    rho: float
    alpha_root: float | None = None  # [rad]
    CL_target: float | None = None
    M: float = 0.0  # informational
    mu: float = 1.7894e-5  # dynamic viscosity [Pa·s], ISA SL

    def __post_init__(self):
        if (self.alpha_root is None) == (self.CL_target is None):
            raise ValueError(
                "FlightCondition: set exactly one of alpha_root or CL_target."
            )

    def q(self) -> float:
        return 0.5 * self.rho * self.V_inf**2

    def Re(self, chord_ref: float) -> float:
        return self.rho * self.V_inf * chord_ref / self.mu


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class LLTResult:
    A: np.ndarray  # Fourier coefficients A_1..A_N
    CL: float
    CD_i: float
    e: float
    delta: float
    alpha_root: float  # [rad]
    wing: WingGeometry
    flight: FlightCondition
    polar: AirfoilPolar
    y_grid: np.ndarray  # uniform spanwise grid [m]
    Gamma: np.ndarray  # Γ on y_grid [m^2/s]
    ell: np.ndarray  # lift per span on y_grid [N/m]
    Cl_local: np.ndarray  # section Cl on y_grid
    CD0: float  # zero-lift profile drag (= airfoil Cd_p(0))

    # ----- callables ------------------------------------------------------

    def Gamma_func(self, y: np.ndarray) -> np.ndarray:
        """Re-evaluate Γ(y) directly from the Fourier series (not interpolated)."""
        return _gamma_from_A(
            self.A, np.asarray(y, dtype=float), self.wing.b, self.flight.V_inf
        )

    def ell_func(self, y: np.ndarray) -> np.ndarray:
        """Spanwise lift per unit span [N/m] via Kutta–Joukowski: ρ V∞ Γ(y)."""
        return self.flight.rho * self.flight.V_inf * self.Gamma_func(y)

    # ----- plotting -------------------------------------------------------

    def plot(self, ax=None, *, show_elliptic: bool = True, normalize: bool = False):
        """Plot the spanwise lift distribution ℓ(y) [N/m].

        Parameters
        ----------
        ax : matplotlib Axes or None
            Existing axes to draw on; a new figure is created if None.
        show_elliptic : bool
            Overlay the elliptic distribution with the same total lift.
        normalize : bool
            If True, plot ℓ(y)/ℓ_root_elliptic vs y/(b/2) (dimensionless).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        y = self.y_grid
        b = self.wing.b
        ell = self.ell

        # Total lift L = ∫ ℓ dy. Elliptic distribution with same L:
        # ℓ_ell(y) = (4 L) / (π b) · √(1 − (2y/b)^2)
        L_total = float(np.trapezoid(ell, y))
        ell_root_ell = 4.0 * L_total / (np.pi * b)
        ell_ell = ell_root_ell * np.sqrt(np.clip(1.0 - (2.0 * y / b) ** 2, 0.0, 1.0))

        if normalize:
            x = 2.0 * y / b
            denom = ell_root_ell if ell_root_ell != 0.0 else 1.0
            ax.plot(x, ell / denom, label="LLT", lw=2)
            if show_elliptic:
                ax.plot(x, ell_ell / denom, "--", label="elliptic (same L)", lw=1)
            ax.set_xlabel(r"$2y/b$")
            ax.set_ylabel(r"$\ell(y) / \ell_{\rm ell,root}$")
        else:
            ax.plot(y, ell, label="LLT", lw=2)
            if show_elliptic:
                ax.plot(y, ell_ell, "--", label="elliptic (same L)", lw=1)
            ax.set_xlabel(r"$y$ [m]")
            ax.set_ylabel(r"$\ell(y)$ [N/m]")

        ax.axhline(0, color="k", lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(
            f"Lift distribution  —  CL={self.CL:.3f},  CD_i={self.CD_i:.4f},  e={self.e:.3f}"
        )
        return ax

    # ----- convenience ----------------------------------------------------

    def summary(self) -> str:
        return (
            f"LLT(N={len(self.A)})  "
            f"alpha_root={np.degrees(self.alpha_root):.3f}°  "
            f"CL={self.CL:.4f}  CD_i={self.CD_i:.5f}  "
            f"CD0={self.CD0:.5f}  e={self.e:.4f}"
        )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_llt(
    wing: WingGeometry,
    polar: AirfoilPolar,
    flight: FlightCondition,
    N: int = 50,
    n_y_grid: int = 200,
) -> LLTResult:
    """Solve Prandtl's LLT and return an LLTResult.

    Parameters
    ----------
    wing : WingGeometry
    polar : AirfoilPolar
        Supplies Cl_alpha and alpha_L0 (overridden per-section if
        wing.alpha_L0_func is provided). Cd_p(Cl) is used to estimate the
        spanwise-averaged profile drag.
    flight : FlightCondition
    N : int
        Number of Fourier modes / collocation points. 50 is plenty for
        smooth planforms; the spec asks for <0.1% change between 50 and 100.
    n_y_grid : int
        Resolution of the uniform y-grid used to evaluate Γ(y), ℓ(y), etc.
        for the downstream NVM stage.
    """
    if N < 2:
        raise ValueError("Need at least N=2 Fourier modes.")

    b = wing.b
    AR = wing.AR
    Cl_alpha = polar.Cl_alpha

    # Cosine-spaced collocation points strictly inside (0, π) to avoid the
    # tip singularity at θ ∈ {0, π}.
    k = np.arange(1, N + 1)
    theta = k * np.pi / (N + 1)
    y_coll = -0.5 * b * np.cos(theta)

    c_coll = wing.chord(y_coll)
    mu = c_coll * Cl_alpha / (4.0 * b)  # μ(θ_k)
    twist = wing.twist(y_coll)
    aL0 = wing.alpha_L0(y_coll, polar.alpha_L0)

    # Linear system M · A = rhs(α_root). Because the equation is linear in
    # α_root, decompose A_n = α_root · A_n^(1) + A_n^(0) and solve once.
    n_idx = np.arange(1, N + 1)
    sin_ntheta = np.sin(np.outer(theta, n_idx))  # [N x N], rows=collocation, cols=mode
    sin_theta = np.sin(theta)[:, None]  # [N x 1]
    M_mat = sin_ntheta * (
        sin_theta + n_idx[None, :] * mu[:, None]
    )  # coefficient matrix

    rhs_unit = mu * sin_theta[:, 0]  # contribution from α_root = 1, aL0=0, twist=0
    rhs_geom = mu * (twist - aL0) * sin_theta[:, 0]  # twist + zero-lift offset

    A_unit = np.linalg.solve(M_mat, rhs_unit)  # A_n^(1)
    A_geom = np.linalg.solve(M_mat, rhs_geom)  # A_n^(0)

    # Trim to alpha_root or CL_target.
    if flight.alpha_root is not None:
        alpha_root = float(flight.alpha_root)
    else:
        # CL = π·AR·A_1 = π·AR·(α·A_unit[0] + A_geom[0]); invert.
        A1_unit, A1_geom = A_unit[0], A_geom[0]
        if abs(A1_unit) < 1e-14:
            raise RuntimeError("Degenerate LLT system: A_1 insensitive to alpha_root.")
        alpha_root = (flight.CL_target / (np.pi * AR) - A1_geom) / A1_unit

    A = alpha_root * A_unit + A_geom

    # Integrated coefficients.
    A1 = A[0]
    CL = float(np.pi * AR * A1)
    if abs(A1) < 1e-14:
        delta, e = 0.0, 1.0
    else:
        higher = np.arange(2, N + 1)
        delta = float(np.sum(higher * (A[1:] / A1) ** 2))
        e = 1.0 / (1.0 + delta)
    CD_i = float(CL**2 / (np.pi * AR) * (1.0 + delta))

    # Evaluate Γ, ℓ, local Cl on a uniform y-grid for the NVM stage.
    y_grid = np.linspace(-b / 2.0, b / 2.0, n_y_grid)
    Gamma = _gamma_from_A(A, y_grid, b, flight.V_inf)
    ell = flight.rho * flight.V_inf * Gamma
    c_grid = wing.chord(y_grid)
    # Section lift coefficient: Cl(y) = 2 Γ(y) / (V∞ · c(y)).
    with np.errstate(divide="ignore", invalid="ignore"):
        Cl_local = np.where(c_grid > 0.0, 2.0 * Gamma / (flight.V_inf * c_grid), 0.0)

    # Wing zero-lift profile drag: assume CD0 == Cd0 from the 2D polar.
    CD0 = float(np.asarray(polar.Cd_p(0.0)))

    return LLTResult(
        A=A,
        CL=CL,
        CD_i=CD_i,
        e=e,
        delta=delta,
        alpha_root=alpha_root,
        wing=wing,
        flight=flight,
        polar=polar,
        y_grid=y_grid,
        Gamma=Gamma,
        ell=ell,
        Cl_local=Cl_local,
        CD0=CD0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gamma_from_A(A: np.ndarray, y: np.ndarray, b: float, V_inf: float) -> np.ndarray:
    """Evaluate Γ(y) = 2·b·V∞·Σ A_n sin(n·θ),  with y = -(b/2)·cos(θ)."""
    y = np.asarray(y, dtype=float)
    # Clip in case of floating drift slightly outside [-b/2, b/2].
    cos_theta = np.clip(-2.0 * y / b, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    n_idx = np.arange(1, len(A) + 1)
    sin_ntheta = np.sin(np.outer(theta, n_idx))  # [n_y, N]
    return 2.0 * b * V_inf * sin_ntheta @ A


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLT smoke test.")
    p.add_argument("airfoil", help="NACA digits (e.g. 2412) or path to .dat")
    p.add_argument("--b", type=float, default=2.5)
    p.add_argument("--AR", type=float, default=6.0)
    p.add_argument("--taper", type=float, default=1.0)
    p.add_argument("--V", type=float, default=20.0)
    p.add_argument("--rho", type=float, default=1.225)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--alpha", type=float, help="root alpha [deg]")
    g.add_argument("--CL", type=float, help="target wing CL")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--plot", action="store_true", help="plot the lift distribution")
    p.add_argument("--normalize", action="store_true", help="normalize plot axes")
    args = p.parse_args()

    from airfoil_polar import get_airfoil_polar

    wing = WingGeometry(b=args.b, AR=args.AR, taper=args.taper)
    flight = FlightCondition(
        V_inf=args.V,
        rho=args.rho,
        alpha_root=np.radians(args.alpha) if args.alpha is not None else None,
        CL_target=args.CL,
    )
    polar = get_airfoil_polar(args.airfoil, Re=flight.Re(wing.c_mean), M=0.0)

    result = solve_llt(wing, polar, flight, N=args.N)
    print(result.summary())
    print(f"  c_root={wing.c_root:.4f} m  c_tip={wing.c_tip:.4f} m  S={wing.S:.4f} m²")

    if args.plot:
        import matplotlib.pyplot as plt

        result.plot(normalize=args.normalize)
        plt.tight_layout()
        plt.show()
