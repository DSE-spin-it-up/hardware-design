"""This file is made to find the cg-position of the fuel tank of the CRJ 1000"""

import numpy as np
import matplotlib.pyplot as plt
from initial_sizing import c


class AirfoilGeometry:
    def __init__(self, file_name="MH112.dat"):
        self.file_name = file_name
        with open(f"airfoils/{self.file_name}", "r") as f:
            lines = f.read().splitlines()
        self.polygon: np.ndarray = np.array(
            [list(map(float, line.split())) for line in lines[1:] if line.strip()],
            dtype=np.float64,
        )
        self.polygon_3d = np.hstack([self.polygon, np.zeros((len(self.polygon), 1))])
        self.y_coords = self.polygon[:, 1]
        self.global_thickness = np.max(self.y_coords) - np.min(self.y_coords)

    def spline(self, poly, x):
        x_coords = poly[:, 0]

        # sort by distance to requested x
        idx_sorted = np.argsort(np.abs(x_coords - x))

        # find first pair with different x-values
        p1 = poly[idx_sorted[0]]

        for idx in idx_sorted[1:]:
            p2 = poly[idx]

            if not np.isclose(p1[0], p2[0]):
                break
        else:
            return p1[1]

        # linear interpolation
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0]

        return a * x + b

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

    def compute_airfoil_area(self):
        p = self.polygon * c
        p_next = np.roll(p, -1, axis=0)

        cross = p[:, 0] * p_next[:, 1] - p_next[:, 0] * p[:, 1]
        return np.abs(0.5 * np.sum(cross))

    def plot_airfoil_geometry(self):
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
        plt.savefig("airfoil.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    airfoil = AirfoilGeometry()
    centroid = airfoil.compute_airfoil_centroid()
    area = airfoil.compute_airfoil_area()
    global_tc = airfoil.global_thickness
    max_tc, max_tc_loc = airfoil.compute_maximum_thickness()
    print("centroid:", centroid)
    print("area:", area)
    print("global thickness to chord ratio:", global_tc)
    print(
        "max thickness to chord ratio:",
        max_tc,
        "at",
        max_tc_loc * 100,
        "percent of the chord",
    )
    print("maximum thickness", max_tc * c * 100, "cm")
    airfoil.plot_airfoil_geometry()
