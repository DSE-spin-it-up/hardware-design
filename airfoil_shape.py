import numpy as np
import matplotlib.pyplot as plt
from initial_sizing import c
import structure as struct

class AirfoilGeometry:

    def __init__(self, airfoil: str):

        self.file_name = airfoil

        with open(self.file_name, "r") as f:
            lines = f.read().splitlines()

        self.polygon: np.ndarray = np.array(
            [list(map(float, line.split())) for line in lines[1:] if line.strip()],
            dtype=np.float64)

        self.polygon_3d = np.hstack([self.polygon, np.zeros((len(self.polygon), 1))])
        self.y_coords = self.polygon[:, 1]
        self.global_thickness = np.max(self.y_coords) - np.min(self.y_coords)

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
        circle1 = plt.Circle((x, y), radius= struct.d / c / 2, color="black", fill=False)
        circle2 = plt.Circle((x, y), radius= ((struct.d / 2) - struct.t) / c , color="black", fill=False)
        centroid = self.compute_airfoil_centroid()
        closed_up = np.vstack([self.polygon, self.polygon[0]])
        ax.plot(closed_up[:, 0], closed_up[:, 1], color="black")
        ax.scatter(centroid[0], centroid[1], color="black")
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.savefig("airfoil.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLT smoke test.")
    p.add_argument("airfoil", help="NACA digits (e.g. 2412) or path to .dat")
    args = p.parse_args()

    airfoil = AirfoilGeometry(args.airfoil)
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
