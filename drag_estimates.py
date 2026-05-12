import numpy as np
import initial_sizing as si
import airfoil_shape as airs
import fuselage as fus

# ------------------------------
# Environmental & flight data
# ------------------------------
R_air = si.R_air
gamma_air = si.gamma_air
T_isa = si.T_isa
V_cruise = si.V_cruise
rho_air = si.rho

# ------------------------------
# Geometry
# ------------------------------
C_wing = si.c            # wing chord
C_tail = si.ct           # tail chord
S_wing = si.Sw
S_tail = si.St
fuselage_length = fus.fuselage_length
fuselage_width = fus.fuselage_width
fuselage_height = fus.fuselage_height
fineness_ratio = fus.fineness

AR_wing = si.AR
CL_wing = si.CL
alpha_wing = np.radians(5)  # [rad]

# ------------------------------
# Airfoil properties
# ------------------------------
# Wing airfoil
wing_airfoil = airs.AirfoilGeometry(file_name="MH112.dat")
max_tc_wing, max_tc_loc_wing = wing_airfoil.compute_maximum_thickness()

# Tail airfoil
tail_airfoil = airs.AirfoilGeometry(file_name="NACA0010.dat")
max_tc_tail, max_tc_loc_tail = tail_airfoil.compute_maximum_thickness()

# ------------------------------
# Flight parameters
# ------------------------------
mu_air = 1.82e-5  # dynamic viscosity
lam_wing = 0           # sweep angle
lam_tail = 0           # sweep angle
Vh_V = 0.85
V_tail = Vh_V * V_cruise  # tail speed (assumed to be lower than wing speed due to downwash)

# Reynolds numbers
Re_wing = rho_air * V_cruise * C_wing / mu_air
Re_tail = rho_air * V_tail * C_tail / mu_air

# Mach number
M_cruise = V_cruise / np.sqrt(R_air * gamma_air * T_isa)
M_tail = Vh_V * M_cruise

# ------------------------------
# Skin friction and form factor
# ------------------------------
# Wing
Cf_wing = 0.455 / (np.log10(Re_wing)**2.58 * (1 + 0.144 * M_cruise**2)**0.65)
FF_wing = (1 + 0.6 / max_tc_loc_wing * max_tc_wing + 100 * max_tc_wing**4) * \
          (1.34 * M_cruise**0.18 * (np.cos(lam_wing))**0.28)
Swet_wing = 2 * S_wing

# Tail
Cf_tail = 0.455 / (np.log10(Re_tail)**2.58 * (1 + 0.144 * M_cruise**2)**0.65)
FF_tail = (1 + 0.6 / max_tc_loc_tail * max_tc_tail + 100 * max_tc_tail**4) * \
          (1.34 * M_tail**0.18 * (np.cos(lam_tail))**0.28)
Swet_tail = 2 * S_tail  # or use actual tail area if different

# ------------------------------
# Fuselage
# ------------------------------
FF_fuselage = 1 + 60 / fineness_ratio**3 + fineness_ratio / 400
Re_fuselage = rho_air * V_cruise * fuselage_length / mu_air
Cf_fuselage = 0.455 / (np.log10(Re_fuselage)**2.58 * (1 + 0.144 * M_cruise**2)**0.65)
Swet_fuselage = 2 * (fuselage_length * fuselage_width + fuselage_length * fuselage_height + fuselage_width * fuselage_height)

# ------------------------------
# Downwash and effective angle
# ------------------------------
downwash_angle = (2 * CL_wing) / (np.pi * AR_wing)
alpha_horizontal = alpha_wing - downwash_angle

# ------------------------------
# Zero-lift drag coefficient
# ------------------------------
CD0 = (Cf_wing * FF_wing * Swet_wing + Cf_fuselage * FF_fuselage * Swet_fuselage + Cf_tail * FF_tail * Swet_tail) / (Swet_fuselage + Swet_wing)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("CD0:", CD0)
    print("Wing wetted area:", Swet_wing)
    print("Tail max thickness:", max_tc_tail, "at", max_tc_loc_tail*100, "% chord")
    print("Fuselage wetted area:", Swet_fuselage)