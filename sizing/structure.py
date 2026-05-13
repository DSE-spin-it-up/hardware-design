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

def calc_t_defl(F, L, defl, E, d):
    return (F * L**3) / (np.pi * E * d**3 * defl)

def calc_t_comp(M, sigma, d):
    return (4 * M) / (np.pi * sigma * d**2)

def calc_mass(L, d, t, rho):
    return ((d / 2) ** 2 - (d / 2 - t) ** 2) * np.pi * L * rho

def calc_defl(F, L, E, I):
    return F * L**3 / (8 * E * I)

cfrp = mat.CFRP()
Y = cfrp.s_c * 0.5
E = cfrp.E
rho = cfrp.rho
sf = 1.2
sigma_lim = Y / 1.2
d_w = tw * 0.8
defl_max = 0.05

M_w = calc_mom_wing(L, bw)
t_w_defl = calc_t_defl(L / 2, bw / 2, defl_max, E, d_w)
t_w_comp = calc_t_comp(M_w, Y, d_w)
t_w = max(t_w_defl, t_w_comp)
if t_w == t_w_defl:
    fail_mod = "Maximum allowable deflection exceeded"
else:
    fail_mod = "Compressive failure"
mass_w = calc_mass(bw, d_w, t_w, rho)

defl_w = calc_defl(L / 2, bw / 2, cfrp.E, calc_I(t_w, d_w))

if __name__ == "__main__":

    print(f"Wing rod outer diameter: {d_w * 1000}mm")
    print(f"Wing rod thickness: {t_w * 1000}mm")
    print(f"Wing rod mass: {mass_w * 1000}g")
    print(f"Failure model: {fail_mod}")
