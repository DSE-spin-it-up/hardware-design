from matplotlib.pylab import radians, sin, cos
import numpy as np
import control_surface_sizing as css

# INPUTS
n=6000/60  # [RPM] Propeller rotational speed
C_p_climb=0.03  # Power coefficient during climb [-]
t_climb=30  # Climb time [s]
# --- Payload & Fleet ---
m_payload = 60  # [kg]
m_drone_empty = 10  # [kg]
n_drones = 3  # [-]

# --- Geometry ---
b = 2.5  # Wingspan [m]
AR = 6.0  # Aspect ratio [-]
t_over_c_root = 0.12  # Root thickness-to-chord ratio [-]
lam = 0.3  # Taper ratio [-]

# --- Aerodynamics ---
Cd0 = 0.045  # Zero-lift drag coefficient [-]
S_payload = 0.25  # Payload frontal area [m^2]
Cd_payload = 1.0  # Payload drag coefficient [-]

# --- Propulsion ---
n_props     = 2   # Number of propellers [-]
eff_motor   = 0.8   # Motor efficiency [-]
eff_prop    = 0.7 # Non-ideal propeller efficiency [-]
T_W_to      = 2.0   # Thrust-to-weight ratio at take-off [-]
J           = 0.47
C_t         = 0.04
D_prop      = 20 * 0.0254

# --- Flight Conditions ---
h_cruise = 300  # Cruise altitude [m]
V_cruise = 20  # Cruise speed [m/s]
climb_rate = 10/3  # Climb rate [m/s]
throttle_climb = 0.9  # Throttle setting during climb [-]

# --- Mission ---
R = 20000  # Range [m]

# --- Battery ---
# specific_energy = 150  # [Wh/kg]
n_cells = 6
voltage_cell = 3.7  # [V]
voltage_battery = n_cells * voltage_cell  # [V]

# --- Materials ---
foam_density = 48  # [kg/m^3]

# --- Atmospheric constants ---
g0 = 9.80665  # [m/s^2]
rho_0 = 1.225  # [kg/m^3] sea-level density
R_air = 287.05  # [J/kg·K] specific gas constant for air

# --- Tail ---
# https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
Vv = 0.04  # Vertical tail volume coefficient [-]
Vh = 0.50  # Horizontal tail volume coefficient [-]
# https://www.fmsg-alling.de/wp-content/uploads/2013/09/V-Leitwerke.pdf
ARt = 6  # Tail aspect ratio [-]
lam_t = 1  # Tail taper ratio [-]
c_cont_to_c_wing = 0.3  # Ratio of control surface chord to wing chord at the same spanwise station [-]
Cl_a_tail=5.72  # Lift curve slope of tail airfoil [-]
Flying_wing = True  # Whether the design is a flying wing (no tail) or not
sweep=15  # Sweep angle of the wing [degrees]
St_Sw=0.25  # Tail area to wing area ratio [-]
Cl_a_wing=5.72  # Lift curve slope of wing airfoil [-]
# DERIVED VARIABLES
#
# --- Mass ---
m_drone_loaded = m_drone_empty + (m_payload / n_drones)  # [kg]

# --- Atmosphere (ISA) ---
T_isa = 15.00 - 0.0065 * h_cruise  # [°C]
p = 101.325e3 * ((T_isa + 273.15) / 288.15) ** 5.2559  # [Pa]
rho = p / (R_air * (T_isa + 273.15))  # [kg/m^3]

# --- Mission Profile ---
t_cruise = R / V_cruise  # [s]

# --- Wing Geometry ---
Sw = b ** 2 / AR  # Wing area [m^2]
c  = Sw / b       # Mean chord [m]
c_root = 2 * Sw / (b * (1 + lam))

# --- Aerodynamics ---
q_cruise = 0.5 * rho * V_cruise**2  # [Pa]
e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64  # [-]
CL = (m_drone_loaded * g0) / (q_cruise * Sw)  # [-]
Cl_airfoil = (AR + 2) * CL / AR  # [-]
if Flying_wing==True:
    Cl_airfoil=Cl_airfoil/cos(radians(sweep))
k = 1 / (np.pi * e * AR)  # [-]
Cd = Cd0 + k * CL**2 + (Cd_payload * S_payload / (n_drones * Sw))  # [-]
L = CL * q_cruise * Sw  # [N]
D = Cd * q_cruise * Sw  # [N]
LD_ratio = L / D  # [-]
# --- Climby ---
C_L_climb = (3*Cd0/k)**(1/2)
V_climb = (2 * m_drone_loaded * g0 / (rho * Sw * C_L_climb))**0.5
Cd_climb = Cd0 + k * C_L_climb**2 + (Cd_payload * S_payload / (n_drones * Sw))  # [-]
D_climb = Cd_climb * 0.5 * rho * V_climb**2 * Sw
T_climb=(D_climb+climb_rate*m_drone_loaded*g0/V_climb)/n_props
P_climb=C_p_climb*rho*(V_climb/J)**3*D_prop**2*n_props

c_t_required=T_climb/(rho*n**2*D_prop**4)
J_climb=V_climb/(n*D_prop)
KV=n*60/throttle_climb/voltage_battery
  

# --- Propulsion ---
thrust_cruise = D
thrust_to = T_W_to * (m_drone_empty * g0)

P_required = D * V_cruise
# D_prop = (((thrust_cruise / n_props) * J ** 2) / (C_t * rho * V_cruise ** 2)) ** (0.5)
thrust_prop = C_t * rho * (V_cruise * D_prop / J) ** 2
thrust_total_prop = thrust_prop * n_props
C_p = (C_t / eff_prop) * J
P_shaft = C_p * rho * ((V_cruise / J) ** 3) * (D_prop ** 2)
rpm_cruise=V_cruise/(J*D_prop)*60
throttle_cruise=rpm_cruise/(KV*voltage_battery)


def motor_weight(KV):
    if KV < 100 or KV > 2000:
        raise ValueError("Value out of range")

    if KV > 100 and KV < 500:
        slope = (900 - 300) / (500 - 100)
        mass = 900 - (KV - 100) * slope

    if KV > 500 and KV < 2000:
        slope = (300 - 50) / (2000 - 500)
        mass = 300 - (KV - 500) * slope

    return mass / 1000  # return in [kg]

motor_weight = motor_weight(KV)
total_motor_weight = motor_weight * n_props

# --- Energy & Battery ---
I_battery = (P_shaft * n_props) / voltage_battery
P_battery = voltage_battery * I_battery
E_cruise     = P_battery * t_cruise +P_climb* t_climb            # [J]


def battery_mass(E_cruise, n_cells, voltage_cell):
    voltage = n_cells * voltage_cell
    C = (E_cruise * 1000 / (3600 * voltage))

    if n_cells == 6:
        a = 0.3988
        b = 0.8810

    mass = a * (C ** b)
    return mass / 1000


# battery_mass = E_cruise / (specific_energy * 3600)  # [kg]
battery_mass = battery_mass(E_cruise, n_cells, voltage_cell)


# --- Structural ---
m_wing = foam_density * (t_over_c_root * c) * Sw  # [kg]

# --- Tail Geometry ---
bh = b * 0.365445026178  # [m] Desmos
Sh = bh**2 / ARt  # [m^2]
L_tail = Vh * Sw * c / Sh  # [m]
Sv = Vv * Sw * b / L_tail  # [m^2]
bv = np.sqrt(2 * ARt * Sw) / 2  # [m]
St = Sh + Sv  # [m^2]
bt = np.sqrt(bh**2 + bv**2)  # [m]
if Flying_wing==False:
    Cmde=Vh*css.tau(c_cont_to_c_wing)*Cl_a_tail
else:
    Cmde=(c+b/2*sin(np.radians(sweep)))/c*St_Sw*css.tau(c_cont_to_c_wing)*Cl_a_wing


def tail_chord(yb):
    return (2 * St) / ((1 + lam_t) * bt) * (1 - (1 - lam_t) * (2 * yb))


ct = tail_chord(0.5)  # [m]
tt = ct * t_over_c_root  # [m]
m_tail = foam_density * St * tt  # [kg]





# DESIGN CRITERIA TESTS

# Testing delivered thrust vs. required thrust
if thrust_total_prop > thrust_cruise:
    print(f"Thrust margin of {thrust_total_prop - thrust_cruise} [N] available.")
else:
    print(f"Additional {thrust_cruise - thrust_total_prop} [N] required")


# OUTPUTS
if __name__ == "__main__":
    print("\n--- Atmosphere ---")
    print(f"  ISA Temperature      : {T_isa:.2f}  °C")
    print(f"  Pressure             : {p:.1f}  Pa")
    print(f"  Air Density          : {rho:.4f}  kg/m³")

    print("\n--- Mass ---")
    print(f"  Loaded drone mass    : {m_drone_loaded:.2f}  kg")
    print(f"  Wing mass            : {m_wing:.3f}  kg")
    print(f"  Tail mass            : {m_tail:.3f}  kg")
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
    print(f"  Total prop. thrust   : {thrust_total_prop:.2f}  N")
    print(f"  Take-off thrust      : {thrust_to:.2f}  N")
    print(f"  Cruise power required : {P_required:.2f}  W")
    print(f"  Propeller Diameter   : {D_prop:.2f}  m")
    print(f"  Shaft Power          : {P_shaft:.2f}  W")
    print(f"  KV Motor             : {KV:.2f}  RPM/V")
    print(f"  Battery Current      : {I_battery:.2f}  A")
    print(f"  Battery Mass         : {battery_mass:.2f}  kg")
    print(f"  Total Motor Mass     : {total_motor_weight:.2f}  kg")

    print("\n--- Energy & Mission ---")
    print(f"  Cruise time          : {t_cruise:.1f}  s  ({t_cruise / 60:.1f} min)")
    print(f"  Cruise energy        : {E_cruise / 3600:.2f}  Wh")

    print("\n--- Tail ---")
    print(f"  Horizontal tail area : {Sh:.4f}  m²")
    print(f"  Vertical tail area   : {Sv:.4f}  m²")
    print(f"  Tail length          : {L_tail:.4f}  m")
    print(f"  Tail chord           : {ct:.4f}  m")
    print(f"  Cmde                 : {Cmde:.4f}  [-]")
    print(f"  c_t_required         : {c_t_required:.4f}  [-]")
    print(f"  J_climb              : {J_climb:.4f}  [-]")
    print(f"  J for cruise            : {J:.2f}  m/s")
    print(f"  Power required       : {P_climb:.2f}  W")
    print(f"  Throttle for Climb   : {throttle_climb:.2f}  [-]")
    print(f"  Throttle for Cruise  : {throttle_cruise:.2f}  [-]")
    print(f"  Climb speed          : {V_climb:.2f}  m/s")
    print(f"  Climb thrust         : {T_climb:.2f}  N")
