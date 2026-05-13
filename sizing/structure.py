import numpy as np
from sizing import initial_sizing as si
import sizing.materials as mat

CL = si.CL  # [-]
q = si.q_cruise  # [kg/ms^2]
Sw = si.Sw  # [m^2]
bw = si.b  # [m]
L = CL * q * Sw  # [N]
cw = si.c  # [m]
tc = si.t_over_c_root  # [-]
tw = cw * tc  # [m]

def calc_mom_wing(F ,L):
    return F * L / 8  # [Nm]

def calc_mom_tail(F, L):
    return F * L  # [Nm]

def calc_I(t, d):
    return (np.pi * t * d ** 3) / 8

def calc_stress(M, t, d):
    Ixx = calc_I(t, d)
    y_max = d / 2  # [m]

    sigma = M * y_max / Ixx  # [Pa]

    return sigma

def calc_d(F, L, E, defl):
    return ((3 * F * L**3) / (np.pi * E * defl))**(1 / 4)

def calc_t(F, L, defl, E, d):
    return (F * L**3) / (np.pi * E * d**3 * defl)

def calc_mass(L, d, t, rho):
    return ((d / 2) ** 2 - (d / 2 - t) ** 2) * np.pi * L * rho

def calc_defl(F, L, E, I):
    return F * L**3 / (8 * E * I)

cfrp = mat.CFRP()
Y = cfrp.s_t * 0.5
E = cfrp.E
rho = cfrp.rho
sf = 1.2
sigma_lim = Y / 1.2
d_max = tw * 0.8
defl_max = 0.05

M_w = calc_mom_wing(L, bw)
d_w = calc_d(L / 2, bw / 2, E, defl_max)
if d_w > d_max:
    d_w = d_max
t_w = calc_t(L / 2, bw / 2, defl_max, E, d_w)
mass_w = calc_mass(bw, d_w, t_w, rho)

defl_w = calc_defl(L / 2, bw / 2, cfrp.E, calc_I(t_w, d_w))

if __name__ == "__main__":

    print(f"Wing rod outer diameter: {d_w * 1000}mm")
    print(f"Wing rod thickness: {t_w * 1000}mm")
    print(f"Wing rod mass: {mass_w * 1000}g")
