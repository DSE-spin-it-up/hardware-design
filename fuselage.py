import numpy as np

b = 3  # Wingspan [m]
AR = 7.5  # Aspect ratio [-]
lam = 1  # Taper ratio [-]

Sw = b ** 2 / AR  # Wing area [m^2]
c  = Sw / b       # Mean chord [m]
c_root = 2 * Sw / (b * (1 + lam))


battery_length=0.212
battery_width=0.090
battery_height=0.060

housing_factor=1.5
casing_factor=1.1

fuselage_length=max(casing_factor*c_root,battery_length * housing_factor*casing_factor)
fuselage_width=battery_width * casing_factor
fuselage_height=battery_height * casing_factor


d_eq=np.sqrt(fuselage_width*fuselage_height)
fineness=fuselage_length/d_eq
print(c_root)
print(fuselage_length)
print(fuselage_width)
print(fuselage_height)
print(fineness)


