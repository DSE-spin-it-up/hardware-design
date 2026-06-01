# Positive elevator deflection (elevator down) produces nose down pitch so Cldeltae is -ve.

import numpy as np

# Elevator
SE_Sh = 0.25     # range: 0.15 - 0.40
bE_bh = 0.08     # range: 0.03 - 0.12

elevator_deflection_up = 25     # deg
elevator_deflection_down = 20   # deg

dedalpha=(2*CLalpha)/(np.pi*AR) # CLalpha is wing lift curve slope, AR is wing aspect ratio
epsilon0=(2*CL0)/(np.pi*AR) # CL0 is wing lift coefficient at 0 deg AoA
epsilon=epsilon0+dedalpha*alpha # alpha is angle of attack in radians
alphah=alpha+ih-epsilon # ih is horizontal tail incidence angle in radians

CLh=(2*Lh)/(rho*V**2*Sh) # Lh is horizontal tail lift, rho is air density, V is free-stream velocity, Sh is horizontal tail area

taue=(CLh-CLalphah*alphah)/(CLalphah*elevator_deflection_up) # CLalphah is horizontal tail lift curve slope, alphah is horizontal tail angle of attack in radians, elevator_deflection_up is elevator deflection in radians

import numpy as np

def cE_ch(taue):
    coeffs = [
        -6.624,
        12.07,
        -8.292,
        3.295,
        0.004942 - taue
    ]

    roots = np.roots(coeffs)

    # Real roots only
    roots = roots[np.isclose(roots.imag, 0)].real

    # Valid aileron area ratio range
    valid = roots[(roots >= 0.2) & (roots <= 0.4)]

    if len(valid) == 0:
        raise ValueError(f"No valid root found for tau_a={tau_a}")

    return valid[0]



CMdeltaE = -CLalphah * etah * Vh  * bE_bh * taue # etah is the dynamic pressure ratio at the horizontal tail so (tail velocity/freestream velocity)^2, Vh is the horizontal tail volume coefficient
CLdeltaE = CLalphah * etah * Vh * Sh_S * bE_bh * taue # CLalphah is horizontal tail lift curve slope, etah is dynamic pressure ratio at the horizontal tail, Vh is horizontal tail volume coefficient, bE_bh is elevator span to horizontal tail span ratio, Sh_S is horizontal tail area to wing area ratio
CLhdeltaE = CLalphah * taue

deltaEcruise = (((T*Z_T)/(q*S*C)+Cm0)*CLalpha+(CLcr-Cl0)*Cmalpha)/(CLalpha*CMdeltaE-Cmalpha*CLdeltaE) # T is thrust, Z_T is vertical distance from thrust action line to which is 2 times wing propeller height plus tail propeller height divided by 3, q is dynamic pressure, S is wing area, C is mean aerodynamic chord, Cm0 is zero-lift pitching moment coefficient, CLcr is lift coefficient during cruise, Cl0 is lift coefficient at 0 deg AoA, Cmalpha is pitching moment slope with respect to angle of attack

# check that maximum positive elevator deflection during cruise is less than elevator_deflection_up, and maximum negative elevator deflection during cruise is less than elevator_deflection_down, otherwise adjust the design parameters accordingly.

