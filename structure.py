import numpy as np
import matplotlib.pyplot as plt
import initial_sizing as si

CL = si.CL  # [-]
q = si.q_cruise  # [kg/ms^2]
Sw = si.Sw  # [m^2]
bw = si.b  # [m]
L = CL * q * Sw  # [N]


cw = si.c  # [m]
tc = si.t_over_c_root  # [-]
tw = cw * tc  # [m]

dens_cfrp = 1.55  # [g/cm^3]
rho_cfrp = dens_cfrp * 10**3  # [kg/m^3]
Y_cfrp = 1000 * 10**6  # [Pa]
t_inch = 1 / 16  # [inch]
t = t_inch * 2.54 / 100  # [m]
d_inch = 1 / 2  # [inch]
d = d_inch * 2.54 / 100  # [m]


M = L * bw / 8  # [Nm]
Ixx = np.pi * t * d**3 / 8  # [m^4]
y_max = d / 2  # [m]

sigma = M * y_max / Ixx  # [Pa]

print(Y_cfrp - sigma)
