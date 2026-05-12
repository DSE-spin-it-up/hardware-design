# tail_drag.py



import numpy as np


def downwash_angle(CL_w, AR_w):
    """
    Simple downwash estimate [rad]
    """

    return (2 * CL_w) / (np.pi * AR_w)


def tail_lift_coefficient(
    alpha,
    incidence,
    CL_alpha_tail,
    downwash,
    alpha_L0=0.0,
):
    """
    Compute actual tail lift coefficient.

    Parameters
    ----------
    alpha : aircraft AoA [rad]
    incidence : tail incidence angle [rad]
    CL_alpha_tail : tail lift slope [1/rad]
    downwash : downwash angle [rad]
    alpha_L0 : zero-lift AoA [rad]
    """

    alpha_tail = alpha + incidence - downwash

    CL_t = CL_alpha_tail * (alpha_tail - alpha_L0)

    return CL_t, alpha_tail


def tail_induced_drag(CL_t, AR_t, e_t=0.9):

    return CL_t**2 / (np.pi * e_t * AR_t)


def tail_profile_drag(Cd0_t=0.01):

    return Cd0_t


def total_tail_drag(
    rho,
    V,
    Sh,
    alpha,
    incidence,
    CL_alpha_tail,
    CL_w,
    AR_w,
    AR_t,
    Cd0_t=0.01,
    e_t=0.9,
    alpha_L0=0.0,
):

    q = 0.5 * rho * V**2

    # Downwash
    eps = downwash_angle(CL_w, AR_w)

    # Tail CL
    CL_t, alpha_tail = tail_lift_coefficient(
        alpha,
        incidence,
        CL_alpha_tail,
        eps,
        alpha_L0,
    )

    # Drag
    CDi_t = tail_induced_drag(CL_t, AR_t, e_t)

    CD_t = Cd0_t + CDi_t

    D_t = q * Sh * CD_t

    return {
        "downwash_angle": eps,
        "alpha_tail": alpha_tail,
        "CL_tail": CL_t,
        "CDi_tail": CDi_t,
        "CD_tail": CD_t,
        "D_tail": D_t,
    }