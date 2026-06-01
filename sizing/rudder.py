# Positive rudder deflection (rudder to the left) yaws nose the left so Cldeltar is -ve.

import numpy as np
from scipy.optimize import fsolve

# Rudder
SR_SV = 0.25     # range: 0.15 - 0.35
cR_CV = 0.28     # range: 0.15 - 0.40
bR_bV = 0.85     # range: 0.70 - 1.00

rudder_deflection_right = np.radians(30)    # deg
rudder_deflection_left = np.radians(30)     # deg

Vgust = 5 # m/s, gust velocity for gust load calculations
CDY = 0.8 # drag coefficient in the y-direction for gust load calculations between 0.5 and 0.8
Kf1 = 0.85 # correction factor for the fuselage contribution to the control derivative (Cnβ ) and is between (0.65 < Kf1 < 0.85)
Kf2 = 1.0 # correction factor the fuselage contribution to the control derivative (Cyβ ) and is between (0.75 < Kf2 < 1)

VT = np.sqrt(Vstall**2+Vgust**2) 

beta = np.arctan(Vgust/Vstall) # sideslip angle during gust

Fw = 0.5 * rho * Vgust**2 * Ss * CDY # aerodynamic force due to gust, CDY is the drag coefficient in the y-direction, # Ss is the side area

def equations(x,
              beta,
              rho, VT, Vgust, S, b,
              Cn0, Cnb, Cndr,
              Cy0, Cyb, Cydr,
              CDY,
              Fw, dc):

    sigma, delta_R = x

    eq1 = (
        0.5 * rho * VT**2 * S * b *
        (Cn0 + Cnb * (beta - sigma) + Cndr * delta_R)
        + Fw * np.cos(dc) * np.cos(sigma)
    )

    eq2 = (
        0.5 * rho * Vgust**2 * S * CDY
        - 0.5 * rho * VT**2 * S *
        (Cy0 + Cyb * (beta - sigma) + Cydr * delta_R)
    )

    return [eq1, eq2]


# Initial guess: [sigma, delta_R]
x0 = [0.0, 0.0]

Cyb = -Kf2 * CLalphav*etav*Sv/S # CLalphav is vertical tail lift curve slope, dsigma_dbeta is the change in sigma with respect to beta, etav is the dynamic pressure ratio at the vertical tail should be same as for horizontal tail, Sv is vertical tail area, S is wing area
Cnb = Kf1 * CLalphav*etav*Sv*lv/(b*S) # CLalphav is vertical tail lift curve slope, etav is dynamic pressure ratio at the vertical tail should be same as for horizontal tail, Sv is vertical tail area, bV is vertical tail span, b is wing span, lv is vertical tail arm which is just the tail length 
Cydr = CLalphav*etav*taur*br/bv*Sr/Sv # CLalphav is vertical tail lift curve slope, etav is dynamic pressure ratio at the vertical tail should be same as for horizontal tail, taur is rudder area to vertical tail area ratio, br is rudder span, bv is vertical tail span, Sr is rudder area, Sv is vertical tail area
Cndr = - CLalphav*Vv*etav*taur*br/bv # CLalphav is vertical tail lift curve slope, Vv is vertical tail volume coefficient, etav is dynamic pressure ratio at the vertical tail should be same as for horizontal tail, taur is rudder area to vertical tail area ratio, br is rudder span, bv is vertical tail span

sol = fsolve(
    equations,
    x0,
    args=(beta,
          rho, VT, Vgust, S, b,
          Cn0, Cnb, Cndr,
          Cy0, Cyb, Cydr,
          CDY,
          Fw, dc)
)

sigma_sol, delta_R_sol = sol
# deltaR is the maximal rudder deflection 30 degrees
print(f"sigma = {sigma_sol}")
print(f"delta_R = {delta_R_sol}")


def cR_cv(taur):
    coeffs = [
        -6.624,
        12.07,
        -8.292,
        3.295,
        0.004942 - taur
    ]

    roots = np.roots(coeffs)

    # Real roots only
    roots = roots[np.isclose(roots.imag, 0)].real

    # Valid aileron area ratio range
    valid = roots[(roots >= 0.2) & (roots <= 0.4)]

    if len(valid) == 0:
        raise ValueError(f"No valid root found for taur={taur}")

    return valid[0]

# if delta_r obtained from system of equation is less than max required to avoid stall in vertical stab, Cr/Cv must be modified and design parameters must be adjusted accordingly.