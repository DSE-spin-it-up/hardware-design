import numpy as np


m_payload = 60  # [kg]
m_drone_empty = 10  # [kg]
n_drones = 3

m_drone_loaded = m_drone_empty + (m_payload / n_drones)
g0 = 9.80665  # [m/s^2]
rho_0 = 1.225  # [kg/m^3]

# Flight Conditions
h_cruise = 300  # [m]
R = 287.05
T = 15.00 - 0.0065 * h_cruise  # [C]
p = (101.325 * ((T + 273.1) / 288.15) ** (5.2559)) * 1000 # [Pa]
rho = p / (R * (T + 273.15))
V_cruise = 20  # [m/s]

# Mission characteristics
R = 20000  # [m]
t_cruise = R / V_cruise

S_payload = 0.25
Cd_payload = 1.0

#Inputs
b = 2.5  # [m]
M = V_cruise / 340
AR = 6


Sw = b**2 / AR
q_cruise = 0.5 * rho * V_cruise ** 2
e = 1.78 * (1 - 0.045 * AR ** (0.68)) - 0.64
CL = (m_drone_loaded * g0) / (q_cruise * Sw)
k = 1 / (np.pi * e * AR)
Cd0 = 0.045   # The drag should be lowered (the 0.6 factor) because we will not have landing gear
Cd = Cd0 + k * CL ** 2 + (Cd_payload * S_payload / n_drones * Sw)
lam = 0.3
c = Sw / b
tc_root = 0.12
N_z = 1.5
Cl_airfoil = (AR + 2) / AR

# Materials
foam_density = 48  # [kg/m3]
m_wing = foam_density * (tc_root * c) * Sw


# m_wing = (0.0038
#               * (N_z * m_drone_loaded * g0) ** 1.06
#               * AR ** 0.38
#               * Sw ** 0.25
#               * (1 + lam) ** 0.21
#               * tc_root ** (-0.14)) / g0


L = CL * q_cruise * Sw
D = Cd * q_cruise * Sw

eff_motor = 0.8
eff_propeller = 0.7
P_cruise = (D * V_cruise) / (eff_motor * eff_propeller)
E_cruise = (P_cruise * t_cruise)  # [J]
specific_energy = 150
Pmax = 4400  # [W]
battery_mass = (1 / specific_energy) * (E_cruise / 3600)


print(f"CL - {CL}")
print(f"Cl airfoil - {Cl_airfoil}")
print(f"Wing Mass - {m_wing}")
