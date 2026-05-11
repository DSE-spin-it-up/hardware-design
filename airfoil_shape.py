"""This file is made to find the cg-position of the fuel tank of the CRJ 1000"""

import numpy as np
import matplotlib.pyplot as plt
from initial_sizing import c


class AirfoilGeometry:
    polygon: np.ndarray = np.array(
        [
            [1.00000000, 0.00000000],
            [0.99649058, 0.00136330],
            [0.98660368, 0.00547709],
            [0.97143035, 0.01199665],
            [0.95144341, 0.02016652],
            [0.92670131, 0.02961565],
            [0.89752683, 0.04030415],
            [0.86449272, 0.05205068],
            [0.82816738, 0.06453411],
            [0.78908692, 0.07737681],
            [0.74774914, 0.09018520],
            [0.70462892, 0.10256493],
            [0.66017070, 0.11411477],
            [0.61473811, 0.12444329],
            [0.56862372, 0.13327372],
            [0.52216328, 0.14043722],
            [0.47572481, 0.14579553],
            [0.42968835, 0.14925105],
            [0.38444902, 0.15070762],
            [0.34032550, 0.15006923],
            [0.29755052, 0.14737924],
            [0.25646503, 0.14284242],
            [0.21755026, 0.13658085],
            [0.18117792, 0.12859862],
            [0.14758049, 0.11898092],
            [0.11697865, 0.10791633],
            [0.08960025, 0.09562226],
            [0.06563801, 0.08233394],
            [0.04525115, 0.06832104],
            [0.02856435, 0.05388872],
            [0.01569521, 0.03937555],
            [0.00671543, 0.02509061],
            [0.00153301, 0.01141669],
            [0.00001242, -0.00096463],
            [0.00024127, -0.00410677],
            [0.00081387, -0.00722284],
            [0.00173454, -0.01004675],
            [0.00255195, -0.01173186],
            [0.00357223, -0.01319148],
            [0.00490827, -0.01446260],
            [0.00659548, -0.01564798],
            [0.00965829, -0.01731512],
            [0.01330108, -0.01886542],
            [0.02415012, -0.02214263],
            [0.04402578, -0.02550975],
            [0.06928121, -0.02743757],
            [0.09968072, -0.02797448],
            [0.13497116, -0.02718773],
            [0.17484399, -0.02520125],
            [0.21893213, -0.02216602],
            [0.26679856, -0.01826931],
            [0.31793754, -0.01370545],
            [0.37179503, -0.00867603],
            [0.42777046, -0.00338197],
            [0.48527553, 0.00198303],
            [0.54373728, 0.00709861],
            [0.60239110, 0.01153371],
            [0.66029295, 0.01496970],
            [0.71646983, 0.01721591],
            [0.76993975, 0.01818017],
            [0.81975105, 0.01787304],
            [0.86499658, 0.01639631],
            [0.90486064, 0.01394373],
            [0.93860949, 0.01072049],
            [0.96545654, 0.00702160],
            [0.98472163, 0.00348974],
            [0.99619815, 0.00093082],
            [1.00000000, 0.00000000],
        ]
    )

    polygon_3d = np.hstack([polygon, np.zeros((len(polygon), 1))])
    y_coords = polygon[:, 1]
    global_thickness = np.max(y_coords) - np.min(y_coords)

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
