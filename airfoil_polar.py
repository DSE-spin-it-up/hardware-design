"""
airfoil_polar.py — 2D airfoil polar generation via XFOIL.

Wraps XFOIL to produce a viscous polar (alpha, Cl, Cd, Cm) for a given
airfoil at a given Reynolds and Mach number, then extracts:

    Cl_alpha   [1/rad]   — lift-curve slope (linear-fit on Cl in [-0.2, 0.6])
    alpha_L0   [rad]     — zero-lift angle of attack
    Cd_p(Cl)             — callable profile-drag interpolator

Results are cached on disk keyed by (airfoil_hash, Re, M, alpha_range)
so repeated calls in an optimization loop are free.

Sign conventions (matches llt_nvm_spec.md §4):
    alpha positive nose-up, Cl positive lift-up.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

CACHE_DIR = Path.home() / ".cache" / "wing_sizing" / "airfoil_polar"
LINEAR_FIT_CL_RANGE = (-0.2, 0.6)
XFOIL_ITER = 200
XFOIL_TIMEOUT_S = 120


@dataclass
class AirfoilPolar:
    """2D viscous polar plus derived linear-region parameters."""

    name: str
    Re: float
    M: float
    alpha: np.ndarray              # [rad]
    Cl: np.ndarray
    Cd: np.ndarray
    Cm: np.ndarray
    Cl_alpha: float                # [1/rad]
    alpha_L0: float                # [rad]
    Cd_p: Callable[[float | np.ndarray], np.ndarray] = field(repr=False)

    def Cl_at(self, alpha: float | np.ndarray) -> np.ndarray:
        """Linearised Cl(alpha) using the fitted slope. Use raw polar for nonlinear regime."""
        return self.Cl_alpha * (np.asarray(alpha) - self.alpha_L0)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        a_deg = np.degrees(self.alpha)
        ax[0].plot(a_deg, self.Cl, "o-", ms=3)
        a_lin = np.linspace(a_deg.min(), a_deg.max(), 50)
        ax[0].plot(a_lin, self.Cl_alpha * (np.radians(a_lin) - self.alpha_L0),
                   "--", lw=1, label=f"linear fit  Cl_α={self.Cl_alpha:.3f} /rad")
        ax[0].axvline(np.degrees(self.alpha_L0), color="k", lw=0.5)
        ax[0].set_xlabel(r"$\alpha$ [deg]"); ax[0].set_ylabel(r"$C_l$"); ax[0].legend()
        ax[1].plot(self.Cl, self.Cd, "o-", ms=3)
        ax[1].set_xlabel(r"$C_l$"); ax[1].set_ylabel(r"$C_d$")
        ax[2].plot(a_deg, self.Cm, "o-", ms=3)
        ax[2].set_xlabel(r"$\alpha$ [deg]"); ax[2].set_ylabel(r"$C_m$")
        for a in ax:
            a.grid(True, alpha=0.3)
        return ax


def get_airfoil_polar(
    airfoil: str | Path,
    Re: float,
    M: float = 0.0,
    alpha_range: tuple[float, float, float] = (-5.0, 15.0, 0.5),
    *,
    use_cache: bool = True,
    xfoil_bin: str = "xfoil",
) -> AirfoilPolar:
    """Run XFOIL and return an AirfoilPolar.

    Parameters
    ----------
    airfoil : str | Path
        Either a NACA designator (e.g. "2412", "NACA2412", "naca 23012")
        or a path to a coordinate `.dat` file in standard Selig format.
    Re : float
        Reynolds number for the viscous solution.
    M : float
        Mach number (compressibility correction inside XFOIL).
    alpha_range : (a_min, a_max, da)
        Sweep bounds in **degrees**.
    use_cache : bool
        If True, cached results are reused when the airfoil contents,
        Re, M and sweep all match.
    xfoil_bin : str
        Name or path of the XFOIL executable.
    """
    spec = _resolve_airfoil(airfoil)
    key = _cache_key(spec, Re, M, alpha_range)
    cache_path = CACHE_DIR / f"{key}.pkl"

    if use_cache and cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    alpha_deg, Cl, Cd, Cm = _run_xfoil_polar(spec, Re, M, alpha_range, xfoil_bin)

    if len(Cl) < 3:
        raise RuntimeError(
            f"XFOIL returned only {len(Cl)} converged points for {spec.name!r} "
            f"at Re={Re:.2e}, M={M}. Try narrowing alpha_range or increasing iterations."
        )

    Cl_alpha, alpha_L0 = _fit_linear_lift(np.radians(alpha_deg), Cl)
    Cd_p = _make_cd_interpolator(Cl, Cd)

    polar = AirfoilPolar(
        name=spec.name,
        Re=Re,
        M=M,
        alpha=np.radians(alpha_deg),
        Cl=Cl,
        Cd=Cd,
        Cm=Cm,
        Cl_alpha=Cl_alpha,
        alpha_L0=alpha_L0,
        Cd_p=Cd_p,
    )

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(polar, f)

    return polar


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _AirfoilSpec:
    """Resolved airfoil: either a NACA digit string or a path to a .dat file."""
    name: str
    naca_digits: str | None
    dat_path: Path | None

    def content_bytes(self) -> bytes:
        if self.naca_digits is not None:
            return f"NACA{self.naca_digits}".encode()
        return self.dat_path.read_bytes()


def _resolve_airfoil(airfoil: str | Path) -> _AirfoilSpec:
    if isinstance(airfoil, Path) or (isinstance(airfoil, str) and airfoil.endswith(".dat")):
        path = Path(airfoil).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return _AirfoilSpec(name=path.stem, naca_digits=None, dat_path=path)

    digits = re.sub(r"[^0-9]", "", str(airfoil))
    if len(digits) not in (4, 5):
        raise ValueError(
            f"Cannot parse airfoil {airfoil!r}: expected a .dat path or a 4/5-digit NACA designator."
        )
    return _AirfoilSpec(name=f"NACA{digits}", naca_digits=digits, dat_path=None)


def _cache_key(spec: _AirfoilSpec, Re: float, M: float, alpha_range: tuple) -> str:
    h = hashlib.sha256()
    h.update(spec.content_bytes())
    h.update(f"|Re={Re:.6e}|M={M:.6f}|a={alpha_range}".encode())
    return f"{spec.name}_{h.hexdigest()[:16]}"


def _run_xfoil_polar(
    spec: _AirfoilSpec,
    Re: float,
    M: float,
    alpha_range: tuple[float, float, float],
    xfoil_bin: str,
):
    if shutil.which(xfoil_bin) is None:
        raise RuntimeError(
            f"XFOIL executable {xfoil_bin!r} not found on PATH. "
            "Install xfoil (e.g. `sudo apt install xfoil`) or pass xfoil_bin=..."
        )

    a_min, a_max, da = alpha_range

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        polar_file = tmp / "polar.dat"

        cmd_lines = ["PLOP", "G F", ""]
        if spec.naca_digits is not None:
            cmd_lines += [f"NACA {spec.naca_digits}"]
        else:
            # XFOIL has a short filename buffer; copy locally and LOAD by basename.
            local_dat = tmp / "airfoil.dat"
            shutil.copyfile(spec.dat_path, local_dat)
            cmd_lines += [f"LOAD {local_dat.name}"]
        cmd_lines += [
            "PANE",
            "OPER",
            f"ITER {XFOIL_ITER}",
            f"VISC {Re:.6e}",
            f"MACH {M:.6f}",
            "PACC",
            str(polar_file),
            "",                                    # no dump file
            f"ASEQ {a_min} {a_max} {da}",
            "PACC",
            "",
            "QUIT",
            "",
        ]
        cmd = "\n".join(cmd_lines) + "\n"

        try:
            subprocess.run(
                [xfoil_bin],
                input=cmd,
                capture_output=True,
                text=True,
                timeout=XFOIL_TIMEOUT_S,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"XFOIL timed out after {XFOIL_TIMEOUT_S}s for {spec.name!r} "
                f"at Re={Re:.2e}, M={M}."
            ) from e

        if not polar_file.exists():
            raise RuntimeError(
                f"XFOIL produced no polar file for {spec.name!r}. "
                "Likely the geometry failed to load or no points converged."
            )

        return _parse_xfoil_polar(polar_file)


def _parse_xfoil_polar(path: Path):
    """Parse an XFOIL PACC polar file. Header ends at the dashed separator line."""
    lines = path.read_text().splitlines()
    data_start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("---"):
            data_start = i + 1
            break
    if data_start is None:
        raise RuntimeError(f"Unrecognised XFOIL polar format in {path}")

    rows = []
    for ln in lines[data_start:]:
        if not ln.strip():
            continue
        parts = ln.split()
        try:
            # XFOIL polar columns: alpha CL CD CDp CM Top_xtr Bot_xtr ...
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[4])])
        except (ValueError, IndexError):
            continue

    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])

    arr = np.array(sorted(rows, key=lambda r: r[0]))
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _fit_linear_lift(alpha_rad: np.ndarray, Cl: np.ndarray) -> tuple[float, float]:
    """Linear least-squares fit Cl = Cl_alpha · (alpha - alpha_L0) on the linear regime."""
    lo, hi = LINEAR_FIT_CL_RANGE
    mask = (Cl >= lo) & (Cl <= hi)
    if mask.sum() < 3:
        # Fall back to the lower half of the polar.
        mask = Cl <= np.median(Cl)
        if mask.sum() < 3:
            raise RuntimeError("Not enough points to fit a lift-curve slope.")

    slope, intercept = np.polyfit(alpha_rad[mask], Cl[mask], 1)
    alpha_L0 = -intercept / slope
    return float(slope), float(alpha_L0)


class _CdInterpolator:
    """Picklable Cd_p(Cl): monotone-cubic inside the sampled range, soft quadratic outside."""

    def __init__(self, Cl: np.ndarray, Cd: np.ndarray):
        order = np.argsort(Cl)
        Cl_s, Cd_s = Cl[order], Cd[order]
        # Drop duplicate Cl values that can appear past stall.
        keep = np.concatenate(([True], np.diff(Cl_s) > 1e-9))
        Cl_s, Cd_s = Cl_s[keep], Cd_s[keep]
        self._pchip = PchipInterpolator(Cl_s, Cd_s, extrapolate=False)
        self._Cl_lo, self._Cl_hi = float(Cl_s[0]), float(Cl_s[-1])
        self._Cd_lo, self._Cd_hi = float(Cd_s[0]), float(Cd_s[-1])

    def __call__(self, cl):
        cl = np.asarray(cl, dtype=float)
        out = self._pchip(cl)
        below = cl < self._Cl_lo
        above = cl > self._Cl_hi
        out = np.where(below, self._Cd_lo + 5.0 * (self._Cl_lo - cl) ** 2, out)
        out = np.where(above, self._Cd_hi + 5.0 * (cl - self._Cl_hi) ** 2, out)
        return out


def _make_cd_interpolator(Cl: np.ndarray, Cd: np.ndarray) -> Callable:
    return _CdInterpolator(Cl, Cd)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description="Generate an XFOIL polar.")
    p.add_argument("airfoil", help="NACA digits (e.g. 2412) or path to .dat")
    p.add_argument("--Re", type=float, default=1e6)
    p.add_argument("--M", type=float, default=0.0)
    p.add_argument("--alpha", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                   default=[-5.0, 15.0, 0.5])
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    polar = get_airfoil_polar(
        args.airfoil, Re=args.Re, M=args.M,
        alpha_range=tuple(args.alpha), use_cache=not args.no_cache,
    )
    print(f"{polar.name}: Cl_alpha = {polar.Cl_alpha:.4f} /rad "
          f"({np.degrees(polar.Cl_alpha):.4f} /deg), "
          f"alpha_L0 = {np.degrees(polar.alpha_L0):.3f} deg")
    polar.plot()
    plt.tight_layout()
    plt.show()
