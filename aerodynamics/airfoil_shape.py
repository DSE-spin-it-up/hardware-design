"""Airfoil geometry utilities."""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def airfoil_thickness_to_chord(airfoil: str | Path) -> float:
    """Maximum t/c of an airfoil specified by a .dat path or NACA digits.

    For NACA 4-/5-digit designators, t/c is the last two digits / 100.
    For .dat files (Selig format), thickness is computed from coordinates.
    """
    s = str(airfoil)
    if s.endswith(".dat"):
        coords = np.loadtxt(s, skiprows=1)
        le = int(np.argmin(coords[:, 0]))
        upper = coords[: le + 1][::-1]  # LE → TE, x ascending
        lower = coords[le:]             # LE → TE, x ascending
        xs = np.linspace(0.01, 0.99, 200)
        y_upper = np.interp(xs, upper[:, 0], upper[:, 1])
        y_lower = np.interp(xs, lower[:, 0], lower[:, 1])
        return float(np.max(y_upper - y_lower))
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) in (4, 5):
        return int(digits[-2:]) / 100.0
    raise ValueError(
        f"Cannot parse airfoil {airfoil!r}: expected .dat path or 4/5-digit NACA."
    )


class AirfoilGeometry:
    """Coordinate-based airfoil geometry loaded from a Selig-format .dat file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open() as f:
            lines = f.read().splitlines()
        self.name = lines[0].strip() if lines else self.path.stem
        self.polygon: np.ndarray = np.array(
            [list(map(float, line.split())) for line in lines[1:] if line.strip()],
            dtype=np.float64,
        )
        self.polygon_3d = np.hstack([self.polygon, np.zeros((len(self.polygon), 1))])
        self.y_coords = self.polygon[:, 1]
        self.global_thickness = float(np.max(self.y_coords) - np.min(self.y_coords))

    def spline(self, poly, x):
        x_coords = poly[:, 0]
        diff = np.abs(x_coords - x)
        diff_sorted = np.sort(diff, axis=0)
        locs_x = np.array(
            [
                np.where(diff == diff_sorted[0])[0][0],
                np.where(diff == diff_sorted[1])[0][0],
            ]
        )
        locs_x = np.sort(locs_x)
        points = np.array([poly[locs_x[0]], poly[locs_x[1]]])
        a = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])
        b = points[0, 1] - a * points[0, 0]
        t = a * x + b
        return t

    def compute_thickness(self, x):
        poly = self.polygon
        le_idx = np.argmin(poly[:, 0])
        poly_upper = poly[: le_idx + 1]
        poly_lower = poly[le_idx:]
        y_upper = self.spline(poly_upper, x)
        y_lower = self.spline(poly_lower, x)
        t = y_upper - y_lower
        return t, y_upper, y_lower

    def compute_maximum_thickness(self):
        x = np.arange(0, 1, 0.0001)
        t = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            t[i], _, _ = self.compute_thickness(x[i])
        t_max = np.max(t)
        t_loc = x[np.where(t == t_max)][0]
        return t_max, t_loc

    def compute_airfoil_centroid(self):
        p = self.polygon
        p_next = np.roll(p, -1, axis=0)

        cross = p[:, 0] * p_next[:, 1] - p_next[:, 0] * p[:, 1]

        area = 0.5 * np.sum(cross)
        cx = np.sum((p[:, 0] + p_next[:, 0]) * cross) / (6.0 * area)
        cy = np.sum((p[:, 1] + p_next[:, 1]) * cross) / (6.0 * area)

        return np.array([cx, cy])

    def compute_airfoil_area(self, chord: float) -> float:
        """Cross-sectional area [m^2] at the given chord length [m]."""
        p = self.polygon * chord
        p_next = np.roll(p, -1, axis=0)

        cross = p[:, 0] * p_next[:, 1] - p_next[:, 0] * p[:, 1]
        return np.abs(0.5 * np.sum(cross))

    def plot_airfoil_geometry(self, out_path: str = "airfoil.png") -> None:
        fig, ax = plt.subplots()
        _, x = self.compute_maximum_thickness()
        t, y_upper, y_lower = self.compute_thickness(x)
        y = (y_upper + y_lower) / 2.0
        circle = plt.Circle((x, y), radius=t / 2, color="black", fill=False)
        centroid = self.compute_airfoil_centroid()
        closed_up = np.vstack([self.polygon, self.polygon[0]])
        ax.plot(closed_up[:, 0], closed_up[:, 1], color="black")
        ax.scatter(centroid[0], centroid[1], color="black")
        ax.add_patch(circle)
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "airfoils/MH112.dat"
    airfoil = AirfoilGeometry(path)
    centroid = airfoil.compute_airfoil_centroid()
    global_tc = airfoil.global_thickness
    max_tc, max_tc_loc = airfoil.compute_maximum_thickness()
    print("airfoil:", airfoil.name)
    print("centroid:", centroid)
    print("global thickness to chord ratio:", global_tc)
    print(
        "max thickness to chord ratio:",
        max_tc,
        "at",
        max_tc_loc * 100,
        "percent of the chord",
    )
    airfoil.plot_airfoil_geometry()
