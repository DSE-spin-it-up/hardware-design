import numpy as np
from sizing import initial_sizing as si

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

def calc_mom_wing(F ,L):
    return F * L / 8  # [Nm]

def calc_mom_tail(F, L):
    return F * L  # [Nm]

def calc_stress(M):
    Ixx = np.pi * t * d ** 3 / 8  # [m^4]
    y_max = d / 2  # [m]

    sigma = M * y_max / Ixx  # [Pa]

    return sigma

def calc_mass(L):
    return ((d / 2) ** 2 - (d / 2 - t) ** 2) * np.pi * L * rho_cfrp


if __name__ == "__main__":
    sigma_w = calc_stress(calc_mom_wing(L, bw))
    mass_w = calc_mass(bw)
    print((0.5 * Y_cfrp - sigma_w) * 10**(-6))
    print(d, t)
    print(mass_w)
