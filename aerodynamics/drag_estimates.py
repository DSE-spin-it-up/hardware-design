import numpy as np
from sizing import initial_sizing as si
from aerodynamics import airfoil_shape as airs

R_AIR = si.R_air
GAMMA_AIR = si.gamma_air
T = si.T_isa
V = si.V_cruise
rho = si.rho
C = si.c
airfoil = airs.AirfoilGeometry("airfoils/MH112.dat")
max_tc, max_tc_loc = airfoil.compute_maximum_thickness()
S = si.Sw

MU = 1.82 * 10**(-5)
LAM = 0
Sw = 2 * S

Re = rho * V * C / MU
M = V / np.sqrt(R_AIR * GAMMA_AIR * T)
Cf = 0.455 / (np.log10(Re)**2.58 * (1 + 0.144 * M**2)**0.65)
FF_wing = (1 + 0.6 / max_tc_loc * max_tc + 100 * max_tc**4) * (1.34 * M**0.18 * (np.cos(LAM))**0.28)

CD0_wing = Cf * FF_wing

if __name__ == "__main__":
    print("CD0 of the wing", CD0_wing)