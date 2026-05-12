"""
wing_lift_curve.py  --  3-D finite-wing / tail lift curve.

Two modes
---------
1. Analytical (fast)
   Uses the Prandtl lifting-line correction formula:

       CL_alpha = Cl_alpha / (1 + Cl_alpha / (pi * AR * e))

   Quick to run; needs an Oswald efficiency guess (e).

2. LLT sweep (accurate)
   Calls llt_solver.solve_llt() at every alpha in a sweep so taper,
   twist and non-elliptic spanwise loading are captured exactly.
   Also returns CD_i(alpha) and e(alpha). Needs llt_solver.py.

Dependencies
------------
    airfoil_polar.py   -- always required
    llt_solver.py      -- required only for --llt / sweep_alpha_llt()
    numpy, matplotlib  (already pulled in by airfoil_polar)

Sign conventions match llt_nvm_spec.md section 4:
    alpha positive nose-up, CL positive lift-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ===========================================================================
# Analytical mode
# ===========================================================================

@dataclass
class WingLiftCurve:
    """3-D finite-wing lift curve via the analytical Prandtl correction."""

    name: str
    airfoil: str
    Re: float
    b: float            # span [m]
    S: float            # reference area [m^2]
    e: float            # Oswald efficiency [-]
    AR: float           # aspect ratio b^2/S
    Cl_alpha_2d: float  # 2-D section slope [1/rad]
    CL_alpha: float     # 3-D wing slope    [1/rad]
    alpha_L0: float     # zero-lift AoA     [rad]

    def CL_at(self, alpha):
        """Linearised 3-D CL(alpha).  alpha in radians."""
        return self.CL_alpha * (np.asarray(alpha, dtype=float) - self.alpha_L0)

    def alpha_at_CL(self, CL):
        """Angle of attack [rad] required for a given 3-D CL."""
        return np.asarray(CL, dtype=float) / self.CL_alpha + self.alpha_L0

    def summary(self):
        print(
            f"\n{'='*52}\n"
            f"  {self.name}  ({self.airfoil}, Re={self.Re:.2e})\n"
            f"{'='*52}\n"
            f"  Span b            {self.b:.3f} m\n"
            f"  Ref. area S       {self.S:.3f} m^2\n"
            f"  Aspect ratio AR   {self.AR:.3f}\n"
            f"  Oswald e          {self.e:.3f}\n"
            f"  Cl_alpha (2D)     {self.Cl_alpha_2d:.4f} /rad"
            f"  ({math.degrees(self.Cl_alpha_2d):.4f} /deg)\n"
            f"  CL_alpha (3D)     {self.CL_alpha:.4f} /rad"
            f"  ({math.degrees(self.CL_alpha):.4f} /deg)\n"
            f"  alpha_L0          {math.degrees(self.alpha_L0):.3f} deg\n"
            f"{'='*52}\n"
        )

    def plot(self, alpha_range_deg=(-5.0, 15.0), ax=None, label=None, color=None):
        """Plot 3-D lift curve alongside the 2-D section."""
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(7, 5))

        a_deg = np.linspace(*alpha_range_deg, 200)
        a_rad = np.radians(a_deg)

        lbl = label or f"{self.name} 3D (AR={self.AR:.2f}, e={self.e})"
        ax.plot(a_deg, self.CL_at(a_rad), lw=2, color=color, label=lbl)
        ax.plot(a_deg, self.Cl_alpha_2d * (a_rad - self.alpha_L0),
                lw=1.2, ls="--", color=color, alpha=0.5,
                label=f"{self.airfoil} 2D section")

        ax.axvline(math.degrees(self.alpha_L0), color="grey", lw=0.8, ls=":")
        ax.axhline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("alpha [deg]")
        ax.set_ylabel("CL")
        ax.set_title(f"Lift curve  --  {self.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


def wing_lift_curve_from_polar(polar, *, b, S, e=0.85, name="Wing"):
    """Build a WingLiftCurve from an AirfoilPolar.

    Parameters
    ----------
    polar : AirfoilPolar   from airfoil_polar.get_airfoil_polar()
    b     : float          span [m]
    S     : float          reference area [m^2]
    e     : float          Oswald efficiency
                             tapered wing  0.85-0.92
                             rectangular   0.75-0.85
                             horiz tail    0.70-0.80
                             T/V-tail      0.65-0.75
    name  : str            label
    """
    AR = b**2 / S
    Cl_a = polar.Cl_alpha
    CL_alpha = Cl_a / (1.0 + Cl_a / (math.pi * AR * e))
    return WingLiftCurve(
        name=name, airfoil=polar.name, Re=polar.Re,
        b=b, S=S, e=e, AR=AR,
        Cl_alpha_2d=Cl_a, CL_alpha=CL_alpha, alpha_L0=polar.alpha_L0,
    )


# ===========================================================================
# LLT sweep mode
# ===========================================================================

@dataclass
class LLTLiftCurve:
    """Result of an alpha sweep through the full LLT solver."""

    name: str
    alpha_deg: np.ndarray
    CL: np.ndarray
    CD_i: np.ndarray
    e_span: np.ndarray
    CL_alpha: float     # fitted slope [1/rad]
    alpha_L0: float     # fitted zero-lift angle [rad]
    results: list       # raw LLTResult objects, one per alpha

    def summary(self):
        print(
            f"\n{'='*52}\n"
            f"  LLT lift curve  --  {self.name}\n"
            f"{'='*52}\n"
            f"  CL_alpha (fitted)  {self.CL_alpha:.4f} /rad"
            f"  ({math.degrees(self.CL_alpha):.4f} /deg)\n"
            f"  alpha_L0 (fitted)  {math.degrees(self.alpha_L0):.3f} deg\n"
            f"  alpha range        {self.alpha_deg[0]:.1f} to {self.alpha_deg[-1]:.1f} deg\n"
            f"  points             {len(self.alpha_deg)}\n"
            f"{'='*52}\n"
        )

    def plot(self, ax=None, label=None, color=None):
        """Plot computed CL(alpha) with linear fit overlaid."""
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(7, 5))

        lbl = label or self.name
        ax.plot(self.alpha_deg, self.CL, "o-", ms=4, lw=2,
                color=color, label=f"{lbl} (LLT)")

        a_fit = np.linspace(self.alpha_deg[0], self.alpha_deg[-1], 200)
        CL_fit = self.CL_alpha * (np.radians(a_fit) - self.alpha_L0)
        ax.plot(a_fit, CL_fit, "--", lw=1.2, color=color, alpha=0.6,
                label=f"linear fit  CL_a={self.CL_alpha:.4f} /rad")

        ax.axvline(math.degrees(self.alpha_L0), color="grey", lw=0.8, ls=":")
        ax.axhline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("alpha [deg]")
        ax.set_ylabel("CL")
        ax.set_title(f"3-D lift curve (LLT)  --  {self.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_drag_polar(self, ax=None, label=None, color=None):
        """Plot induced drag polar CD_i vs CL."""
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 5))

        lbl = label or self.name
        ax.plot(self.CL, self.CD_i, "o-", ms=4, lw=2, color=color, label=lbl)
        ax.set_xlabel("CL")
        ax.set_ylabel("CD_i")
        ax.set_title(f"Induced drag polar  --  {self.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


def sweep_alpha_llt(
    wing,
    polar,
    flight_base,
    alpha_range_deg=(-15.0, 25.0, 1.0),
    *,
    N=50,
    n_y_grid=200,
    linear_CL_range=(-0.5, 0.9),
    name="Wing",
):
    """Sweep alpha through solve_llt() and return an LLTLiftCurve.

    Parameters
    ----------
    wing            : WingGeometry    from llt_solver.py
    polar           : AirfoilPolar    from airfoil_polar.py
    flight_base     : FlightCondition V_inf/rho/M/mu used;
                      alpha_root is overridden at each sweep point
    alpha_range_deg : (min, max, step) degrees
    N               : LLT Fourier modes
    n_y_grid        : spanwise grid points
    linear_CL_range : (lo, hi) CL window for slope fit
    name            : surface label
    """
    try:
        from llt_solver import solve_llt, FlightCondition
    except ImportError as exc:
        raise ImportError(
            "llt_solver.py must be in the same folder or on PYTHONPATH."
        ) from exc

    a_min, a_max, da = alpha_range_deg
    alpha_deg_arr = np.arange(a_min, a_max + 1e-9, da)

    CL_arr, CDi_arr, e_arr, results = [], [], [], []

    for a_deg in alpha_deg_arr:
        fc = FlightCondition(
            V_inf=flight_base.V_inf,
            rho=flight_base.rho,
            M=flight_base.M,
            mu=flight_base.mu,
            alpha_root=math.radians(a_deg),
        )
        res = solve_llt(wing, polar, fc, N=N, n_y_grid=n_y_grid)
        CL_arr.append(res.CL)
        CDi_arr.append(res.CD_i)
        e_arr.append(res.e)
        results.append(res)

    CL_arr  = np.array(CL_arr)
    CDi_arr = np.array(CDi_arr)
    e_arr   = np.array(e_arr)

    alpha_rad_arr = np.radians(alpha_deg_arr)
    cl_lo, cl_hi = linear_CL_range
    mask = (CL_arr >= cl_lo) & (CL_arr <= cl_hi)
    if mask.sum() < 3:
        mask = np.ones(len(CL_arr), dtype=bool)
    slope, intercept = np.polyfit(alpha_rad_arr[mask], CL_arr[mask], 1)
    alpha_L0_fit = -intercept / slope

    return LLTLiftCurve(
        name=name,
        alpha_deg=alpha_deg_arr,
        CL=CL_arr,
        CD_i=CDi_arr,
        e_span=e_arr,
        CL_alpha=float(slope),
        alpha_L0=float(alpha_L0_fit),
        results=results,
    )


def compare_surfaces(*surfaces, alpha_range_deg=(-5.0, 15.0)):
    """Overlay multiple WingLiftCurve or LLTLiftCurve objects on one plot."""
    import matplotlib.pyplot as plt
    _fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, surf in enumerate(surfaces):
        surf.plot(ax=ax, color=colors[i % len(colors)])
    ax.set_title("3-D lift curves  --  surface comparison")
    ax.legend()
    plt.tight_layout()
    return ax


# ===========================================================================
# CLI  (only runs when you call: python wing_lift_curve.py ...)
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    try:
        from airfoil_polar import get_airfoil_polar
    except ImportError as exc:
        raise SystemExit(
            "airfoil_polar.py not found.\n"
            "Run from the same folder as airfoil_polar.py."
        ) from exc

    p = argparse.ArgumentParser(
        description=(
            "3-D finite-wing lift curve.\n"
            "  Default  : analytical Prandtl correction (fast).\n"
            "  --llt    : full LLT alpha sweep (accurate, needs llt_solver.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("airfoil",          help="NACA digits e.g. 2412, or path to .dat")
    p.add_argument("--Re",   type=float, default=4e5,   help="Reynolds number")
    p.add_argument("--M",    type=float, default=0.0,   help="Mach number")
    p.add_argument("--V",    type=float, default=20.0,  help="Free-stream speed m/s  (LLT only)")
    p.add_argument("--rho",  type=float, default=1.225, help="Air density kg/m3      (LLT only)")
    p.add_argument("--b",    type=float, required=True, help="Span [m]")
    p.add_argument("--S",    type=float, required=True, help="Reference area [m^2]")
    p.add_argument("--taper",type=float, default=1.0,   help="Taper ratio            (LLT only)")
    p.add_argument("--e",    type=float, default=0.85,  help="Oswald efficiency      (analytical only)")
    p.add_argument("--name", type=str,   default="Wing",help="Surface label")
    p.add_argument("--llt",  action="store_true",       help="Use full LLT alpha sweep")
    p.add_argument("--N",    type=int,   default=50,    help="LLT Fourier modes      (LLT only)")
    p.add_argument("--alpha", nargs=3, type=float,
                   metavar=("MIN", "MAX", "STEP"), default=[-5.0, 15.0, 1.0],
                   help="Alpha sweep [deg]  default: -5 15 1")
    p.add_argument("--cache", action="store_true", help="Cache XFOIL polar to disk")
    args = p.parse_args()

    polar = get_airfoil_polar(
        args.airfoil,
        Re=args.Re,
        M=args.M,
        alpha_range=(-8.0, 18.0, 0.5),
        use_cache=args.cache,
    )

    if args.llt:
        try:
            from llt_solver import WingGeometry, FlightCondition
        except ImportError as exc:
            raise SystemExit("llt_solver.py not found on PYTHONPATH.") from exc

        wing        = WingGeometry(b=args.b, S=args.S, taper=args.taper)
        flight_base = FlightCondition(V_inf=args.V, rho=args.rho, alpha_root=0.0)

        lc = sweep_alpha_llt(
            wing, polar, flight_base,
            alpha_range_deg=tuple(args.alpha),
            N=args.N,
            name=args.name,
        )
        lc.summary()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        lc.plot(ax=axes[0])
        lc.plot_drag_polar(ax=axes[1])

    else:
        wlc = wing_lift_curve_from_polar(
            polar, b=args.b, S=args.S, e=args.e, name=args.name
        )
        wlc.summary()
        fig, ax = plt.subplots(figsize=(7, 5))
        wlc.plot(ax=ax, alpha_range_deg=(args.alpha[0], args.alpha[1]))

    plt.tight_layout()
    plt.show()