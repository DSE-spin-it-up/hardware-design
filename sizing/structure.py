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
Y_cfrp = 3000 * 10**6  # [Pa]
t_inch = 1 / 16  # [inch]
t = t_inch * 2.54 / 100  # [m]
d_inch = 1 / 2  # [inch]
d = d_inch * 2.54 / 100  # [m]

class Material:
    def __init__(self, E, rho, Y, s_t):
        self.E = E
        self.rho = rho
        self.Y = Y
        self.s_t = s_t

class CFRP(Material):  # https://www.researchgate.net/publication/342938721_Experimental_Investigation_of_Reinforced_Concrete_Beam_with_Openings_Strengthened_Using_FRP_Sheets_under_Cyclic_Load
    def __init__(self):
        super().__init__(
            E=230e9,
            rho=1.72e3,
            Y=3400e6,
            s_t=3400e6
        )

class EPP(Material):  # https://www.foambymail.com/polypropylene-foam-sheet.html?srsltid=AfmBOorBTKilq9ebl6LLzYihjew-KWV77s8RLA2MVI6mPx15FnjyM8NA
    def __init__(self):
        super().__init__(
            E=230e9,
            rho=20.824002386148,
            Y=262e3,
            s_t=262e3
        )

class CF_PLA(Material):  # https://www.iemai3d.com/wp-content/uploads/2020/12/CF-PLA_TDS.pdf
    def __init__(self):
        super().__init__(
            E=4950e6,
            rho=1.29e3,
            Y= 48e6,
            s_t = 48e6
        )

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
