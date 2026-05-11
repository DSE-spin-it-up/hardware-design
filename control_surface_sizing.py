import numpy as np
import scipy as sci
import initial_sizing as ins

# Assumed parameters

cl_a = 2 * np.pi
cd0 = 0.04
tau = 0.3
S_ref = ins.Sw
b = ins.b
lam = ins.lam
c_root = ins.c_root
V_cruise = ins.V_cruise

def c_at_y(c_root, lam):
    def c_at_y_eval(y_frac):
        # y_frac defined as y / (b/2)
        c = c_root * (1 - (1 - lam)) * y_frac
        return c
    return c_at_y_eval


def tau(c_cont_to_c_wing):
   if c_cont_to_c_wing > 0.7:
       raise ValueError("Value to high!")

   if c_cont_to_c_wing < 0.2:
       tau = 2 * c_cont_to_c_wing

   else:
       tau = 0.4 + (0.4 / 0.5) * c_cont_to_c_wing

   return tau


def compute_cl_da(start_y_frac, end_y_frac, c_at_y, b, Sref, cl_a, tau):
    c_integral = sci.integrate.quad(lambda y: c_at_y(y) * y * (b / 2) ** 2, start_y_frac, end_y_frac)[0]
    cl_da = ((2 * cl_a * tau) / (Sref * b)) * c_integral
    return cl_da


def compute_cl_p(cl_a, cd0, Sref, b, c_at_y):
    c_integral = sci.integrate.quad(lambda y: c_at_y(y) * (y ** 2) * (b / 2) ** 3, 0, 1)[0]
    c_l_p = - (4 * (cl_a + cd0) / (Sref * b ** 2)) * c_integral
    return c_l_p

def optimize_aileron_span(roll_req, max_da, max_y_frac, V_cruise, b, compute_cl_p, compute_cl_da, c_at_y, c_cont_to_c_wing):
    start_y_frac = max_y_frac
    cl_da = compute_cl_da(start_y_frac, max_y_frac, c_at_y_eval, b, S_ref, cl_a, tau(c_cont_to_c_wing))
    cl_p = compute_cl_p(cl_a, cd0, S_ref, b, c_at_y_eval)
    P = -(cl_da / cl_p) * max_da * (2 * V_cruise / b)
    while P < roll_req:
        start_y_frac -= 0.01
        cl_da = compute_cl_da(start_y_frac, max_y_frac, c_at_y_eval, b, S_ref, cl_a, tau(c_cont_to_c_wing))
        P = -(cl_da / cl_p) * max_da * (2 * V_cruise / b)
    return start_y_frac

c_at_y_eval = c_at_y(c_root, lam)

roll_req = np.deg2rad(60)
max_da = np.deg2rad(10)
max_y_frac = 0.8
c_cont_to_c_wing = 0.3

start_y_frac = optimize_aileron_span(roll_req, max_da, max_y_frac, V_cruise, b, compute_cl_p, compute_cl_da, c_at_y, c_cont_to_c_wing)
print(start_y_frac)
