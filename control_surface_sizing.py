import numpy as np
import scipy as sci

# Assumed parameters (defaults; do not import initial_sizing here to avoid circular imports)
cl_a = 2 * np.pi
cd0 = 0.04

def c_at_y(c_root, lam):
    def c_at_y_eval(y_frac):
        # y_frac defined as y / (b/2)
        c = c_root * (1 - (1 - lam)) * y_frac
        return c
    return c_at_y_eval


def tau(c_cont_to_c_wing):
    if c_cont_to_c_wing > 0.7:
        raise ValueError("Value too high!")

    if c_cont_to_c_wing < 0.2:
        val = 2 * c_cont_to_c_wing

    else:
        val = 0.4 + (0.4 / 0.5) * c_cont_to_c_wing

    return val


def compute_cl_da(start_y_frac, end_y_frac, c_at_y_func, b, Sref, cl_a_val, tau_val):
    c_integral = sci.integrate.quad(lambda y: c_at_y_func(y) * y * (b / 2) ** 2, start_y_frac, end_y_frac)[0]
    cl_da = ((2 * cl_a_val * tau_val) / (Sref * b)) * c_integral
    return cl_da


def compute_cl_p(cl_a_val, cd0_val, Sref, b, c_at_y_func):
    c_integral = sci.integrate.quad(lambda y: c_at_y_func(y) * (y ** 2) * (b / 2) ** 3, 0, 1)[0]
    c_l_p = - (4 * (cl_a_val + cd0_val) / (Sref * b ** 2)) * c_integral
    return c_l_p


def optimize_aileron_span(roll_req, max_da, max_y_frac, V_cruise, b, compute_cl_p_func, compute_cl_da_func, c_at_y_func, c_cont_to_c_wing, Sref, cl_a_val=cl_a, cd0_val=cd0):
    start_y_frac = max_y_frac
    cl_da = compute_cl_da_func(start_y_frac, max_y_frac, c_at_y_func, b, Sref, cl_a_val, tau(c_cont_to_c_wing))
    cl_p = compute_cl_p_func(cl_a_val, cd0_val, Sref, b, c_at_y_func)
    P = -(cl_da / cl_p) * max_da * (2 * V_cruise / b)
    while P < roll_req:
        start_y_frac -= 0.01
        cl_da = compute_cl_da_func(start_y_frac, max_y_frac, c_at_y_func, b, Sref, cl_a_val, tau(c_cont_to_c_wing))
        P = -(cl_da / cl_p) * max_da * (2 * V_cruise / b)
    return start_y_frac


if __name__ == "__main__":
    # Example usage when executed directly (keeps module import-safe)
    c_root = 1.0
    lam = 0.3
    b = 2.5
    S_ref = b ** 2 / 6.0
    V_cruise = 20.0
    c_at_y_eval = c_at_y(c_root, lam)
    roll_req = np.deg2rad(60)
    max_da = np.deg2rad(10)
    max_y_frac = 0.8
    c_cont_to_c_wing = 0.3
    start_y_frac = optimize_aileron_span(roll_req, max_da, max_y_frac, V_cruise, b, compute_cl_p, compute_cl_da, c_at_y_eval, c_cont_to_c_wing, S_ref)
    print(start_y_frac)
