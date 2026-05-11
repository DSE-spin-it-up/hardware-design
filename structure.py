import numpy as np
import matplotlib.pyplot as plt
import initial_sizing as si

CL = si.CL # [-]
q = si.q_cruise # [kg/ms^2]
Sw = si.Sw # [m^2]
L = CL * q * Sw # [N]

cw = si.c # [m]
tc = si.t_over_c_root # [-]
tw = cw * tc # [m]

dens_cfrp = 1.55 # [g/cm^3]
rho_cfrp = dens_cfrp * 10**3 # [kg/m^3]