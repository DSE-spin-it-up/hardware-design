# Positive rudder deflection (rudder to the left) yaws nose the left so Cldeltar is -ve.

import numpy as np

# Rudder
SR_SV = 0.25     # range: 0.15 - 0.35
cR_CV = 0.28     # range: 0.15 - 0.40
bR_bV = 0.85     # range: 0.70 - 1.00

rudder_deflection_right = 30    # deg
rudder_deflection_left = 30     # deg

Vgust = 5 # m/s, gust velocity for gust load calculations

VT = np.sqrt(Vstall**2+Vgust**2) 
