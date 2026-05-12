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

# --- Setup (done once) ---
wing = WingGeometry(b=b, AR=AR, taper=taper)
polar = get_airfoil_polar("airfoils/MH112.dat", Re=Re, M=M)

res = 0.5
alpha_values = np.arange(-5, 15 + res, res)  # degrees
# --- Sweep ---
results: list[LLTResult] = []
for alpha_deg in alpha_values:
    flight = FlightCondition(
        V_inf=V,
        rho=rho,
        alpha_root=np.radians(alpha_deg),
    )
    result = solve_llt(wing, polar, flight, N=50)
    results.append(result)

# --- Extract arrays for post-processing ---
CL_arr  = np.array([r.CL    for r in results])
CDi_arr = np.array([r.CD_i  for r in results])
e_arr   = np.array([r.e     for r in results])

dα = np.radians(res)  # convert step to radians
CL_alpha_arr = np.gradient(CL_arr, dα)  # dCL/dα at every alpha [1/rad]

# LLT is linear, so the whole range should fit well — adjust mask if needed
mask = (alpha_values >= 0) & (alpha_values <= 10)
alpha_rad = np.radians(alpha_values)

coeffs = np.polyfit(alpha_rad[mask], CL_arr[mask], deg=1)
CL_alpha = coeffs[0]        # lift slope [1/rad]
alpha_L0 = -coeffs[1] / coeffs[0]  # zero-lift angle [rad]

print(f"CL_alpha = {CL_alpha:.4f} /rad  ({np.radians(1)*CL_alpha:.4f} /deg)")
print(f"alpha_L0 = {np.degrees(alpha_L0):.2f} deg")


if __name__ == "__main__":

    if __name__ == "__main__":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(alpha_values, CL_arr)
        ax1.plot(alpha_values, np.polyval(coeffs, np.radians(alpha_values)),
                 "--", label=f"fit: CL_α={CL_alpha:.3f}/rad")
        ax1.set_xlabel("α [deg]")
        ax1.set_ylabel("CL")
        ax1.legend()

        ax2.plot(alpha_values, np.degrees(CL_alpha_arr))  # /deg for readability
        ax2.set_xlabel("α [deg]")
        ax2.set_ylabel("dCL/dα [1/deg]")
        ax2.axhline(np.degrees(CL_alpha), color="r", linestyle="--", label="fit value")
        ax2.legend()

        plt.tight_layout()
        plt.show()