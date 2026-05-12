# run_sweep.py
import numpy as np
from matplotlib import pyplot as plt

from llt_solver import WingGeometry, FlightCondition, LLTResult, solve_llt
from airfoil_polar import get_airfoil_polar
import initial_sizing as si
import drag_estimates as dr

b = si.b
AR = si.AR
taper = si.lam
V = si.V_cruise
rho = si.rho
Re = dr.Re_wing
M = dr.M

def create_3d_curve(airfoil):
    wing = WingGeometry(b=b, AR=AR, taper=taper)
    polar = get_airfoil_polar(airfoil, Re=Re, M=M)

    res = 0.5
    alpha_values = np.arange(-5, 15 + res, res)

    results: list[LLTResult] = []
    for alpha_deg in alpha_values:
        flight = FlightCondition(
            V_inf=V,
            rho=rho,
            alpha_root=np.radians(alpha_deg),
        )
        result = solve_llt(wing, polar, flight, N=50)
        results.append(result)

    CL_arr  = np.array([r.CL    for r in results])

    dα = np.radians(res)
    CL_alpha_arr = np.gradient(CL_arr, dα)  # dCL/dα at every alpha [1/rad]

    # LLT is linear, so the whole range should fit well — adjust mask if needed
    mask = (alpha_values >= 0) & (alpha_values <= 10)
    alpha_rad = np.radians(alpha_values)

    coeffs = np.polyfit(alpha_rad[mask], CL_arr[mask], deg=1)
    CL_alpha = coeffs[0]        # lift slope [1/rad]

    return alpha_values, CL_arr, coeffs, CL_alpha, CL_alpha_arr


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLT smoke test.")
    p.add_argument("airfoil", help="NACA digits (e.g. 2412) or path to .dat")
    args = p.parse_args()

    alpha_values, CL_arr, coeffs, CL_alpha, CL_alpha_arr = create_3d_curve(args.airfoil)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(alpha_values, CL_arr)
    ax.plot(alpha_values, np.polyval(coeffs, np.radians(alpha_values)),
             "--", label=f"fit: CL_α={CL_alpha:.3f}/rad")
    ax.set_xlabel("α [deg]")
    ax.set_ylabel("CL")
    ax.legend()

    plt.show()