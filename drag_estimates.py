import numpy as np
import initial_sizing as si
import airfoil_shape as airs
import fuselage as fus

R_AIR = si.R_air
GAMMA_AIR = si.gamma_air
T = si.T_isa
V = si.V_cruise
rho = si.rho
C = si.c
airfoil = airs.AirfoilGeometry()
max_tc, max_tc_loc = airfoil.compute_maximum_thickness()
S = si.Sw
L = fus.fuselage_length
W = fus.fuselage_width
H = fus.fuselage_height

MU = 1.82 * 10**(-5)
LAM = 0

Re_wing = rho * V * C / MU
M = V / np.sqrt(R_AIR * GAMMA_AIR * T)
Cf_wing = 0.455 / (np.log10(Re_wing)**2.58 * (1 + 0.144 * M**2)**0.65)
FF_wing = (1 + 0.6 / max_tc_loc * max_tc + 100 * max_tc**4) * (1.34 * M**0.18 * (np.cos(LAM))**0.28)
Sw_wing = 2 * S

fin = fus.fineness
FF_fus = 1 + 60 / fin**3 + fin / 400
Re_fus = rho * V * L / MU
Cf_fus = 0.455 / (np.log10(Re_wing)**2.58 * (1 + 0.144 * M**2)**0.65)
Sw_fus = 2 * (L * W + L * H + W * H)



CD0 = 1 / (Sw_fus + Sw_wing) * (Cf_wing * FF_wing * Sw_wing + Cf_fus * FF_fus * Sw_fus)


if __name__ == "__main__":
    print("CD0", CD0)