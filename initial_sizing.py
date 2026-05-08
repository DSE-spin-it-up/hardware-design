import numpy as np

# INPUTS

# --- Payload & Fleet ---
m_payload       = 60        # [kg]
m_drone_empty   = 10        # [kg]
n_drones        = 3         # [-]

# --- Geometry ---
b       = 2.5   # Wingspan [m]
AR      = 6.0   # Aspect ratio [-]
t_over_c_root = 0.12  # Root thickness-to-chord ratio [-]
lam     = 0.3   # Taper ratio [-]

# --- Aerodynamics ---
Cd0         = 0.045  # Zero-lift drag coefficient [-]
S_payload   = 0.25   # Payload frontal area [m^2]
Cd_payload  = 1.0    # Payload drag coefficient [-]

# --- Propulsion ---
n_props     = 2     # Number of propellers [-]
D_prop      = 0.3   # Propeller diameter [m]
eff_motor   = 0.8   # Motor efficiency [-]
eff_nonideal = 0.9  # Non-ideal propeller efficiency [-]
T_W_to      = 2.0   # Thrust-to-weight ratio at take-off [-]

# --- Flight Conditions ---
h_cruise  = 300   # Cruise altitude [m]
V_cruise  = 20    # Cruise speed [m/s]

# --- Mission ---
R = 20000  # Range [m]

# --- Battery ---
specific_energy = 150  # [Wh/kg]

# --- Materials ---
foam_density = 48  # [kg/m^3]

# --- Atmospheric constants ---
g0    = 9.80665  # [m/s^2]
rho_0 = 1.225    # [kg/m^3] sea-level density
R_air = 287.05   # [J/kg·K] specific gas constant for air


# DERIVED VARIABLES
#
# --- Mass ---
m_drone_loaded = m_drone_empty + (m_payload / n_drones)  # [kg]

# --- Atmosphere (ISA) ---
T_isa = 15.00 - 0.0065 * h_cruise                          # [°C]
p     = 101.325e3 * ((T_isa + 273.15) / 288.15) ** 5.2559  # [Pa]
rho   = p / (R_air * (T_isa + 273.15))                     # [kg/m^3]

# --- Mission Profile ---
t_cruise = R / V_cruise  # [s]

# --- Wing Geometry ---
Sw = b ** 2 / AR  # Wing area [m^2]
c  = Sw / b       # Mean chord [m]

# --- Aerodynamics ---
q_cruise   = 0.5 * rho * V_cruise ** 2                                         # [Pa]
e          = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64                            # [-]
CL         = (m_drone_loaded * g0) / (q_cruise * Sw)                           # [-]
Cl_airfoil = (AR + 2) * CL / AR                                                # [-]
k          = 1 / (np.pi * e * AR)                                              # [-]
Cd         = Cd0 + k * CL ** 2 + (Cd_payload * S_payload / (n_drones * Sw))    # [-]
L          = CL * q_cruise * Sw                                                # [N]
D          = Cd * q_cruise * Sw                                                # [N]
LD_ratio   = L / D                                                             # [-]

# --- Propulsion ---
A_prop        = (np.pi / 4) * D_prop ** 2
thrust_cruise = D
thrust_to     = T_W_to * (m_drone_empty * g0)

P_cruise = ((thrust_cruise / n_props) ** 1.5
            / (np.sqrt(2 * rho * A_prop) * eff_nonideal)) / eff_motor
P_to     = ((thrust_to / n_props) ** 1.5
            / (np.sqrt(2 * rho * A_prop) * eff_nonideal)) / eff_motor

# --- Energy & Battery ---
E_cruise     = P_cruise * t_cruise                  # [J]
battery_mass = E_cruise / (specific_energy * 3600)  # [kg]

# --- Structural ---
m_wing = foam_density * (t_over_c_root * c) * Sw  # [kg]


print("\n--- Atmosphere ---")
print(f"  ISA Temperature      : {T_isa:.2f}  °C")
print(f"  Pressure             : {p:.1f}  Pa")
print(f"  Air Density          : {rho:.4f}  kg/m³")

print("\n--- Mass ---")
print(f"  Loaded drone mass    : {m_drone_loaded:.2f}  kg")
print(f"  Wing mass            : {m_wing:.3f}  kg")
print(f"  Battery mass         : {battery_mass:.3f}  kg")

print("\n--- Wing Geometry ---")
print(f"  Wing area            : {Sw:.4f}  m²")
print(f"  Mean chord           : {c:.4f}  m")

print("\n--- Aerodynamics ---")
print(f"  Dynamic pressure     : {q_cruise:.2f}  Pa")
print(f"  Oswald efficiency    : {e:.4f}")
print(f"  CL (drone)           : {CL:.4f}")
print(f"  Cl (airfoil)         : {Cl_airfoil:.4f}")
print(f"  CD                   : {Cd:.5f}")
print(f"  L/D ratio            : {LD_ratio:.2f}")
print(f"  Lift                 : {L:.2f}  N")
print(f"  Drag                 : {D:.2f}  N")

print("\n--- Propulsion ---")
print(f"  Cruise thrust        : {thrust_cruise:.2f}  N")
print(f"  Take-off thrust      : {thrust_to:.2f}  N")
print(f"  Cruise power         : {P_cruise:.2f}  W")
print(f"  Take-off power       : {P_to:.2f}  W")
print(f"  Propeller Diameter   : {D_prop:.2f}  m")

print("\n--- Energy & Mission ---")
print(f"  Cruise time          : {t_cruise:.1f}  s  ({t_cruise/60:.1f} min)")
print(f"  Cruise energy        : {E_cruise/3600:.2f}  Wh")
