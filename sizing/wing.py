from dataclasses import dataclass

import numpy as np

# Atmospheric constants (not design parameters).
g0 = 9.80665       # [m/s^2]
rho_0 = 1.225      # [kg/m^3] sea-level density
R_air = 287.05     # [J/kg·K]
gamma_air = 1.4    # [-]


@dataclass
class SizingInputs:
    # Payload & Fleet
    m_payload: float = 60        # [kg]
    m_drone_empty: float = 10    # [kg]
    n_drones: int = 3            # [-]
    # Geometry
    b: float = 2.5               # Wingspan [m]
    AR: float = 6.0              # Aspect ratio [-]
    lam: float = 0.3             # Taper ratio [-]
    # Aerodynamics
    Cd0: float = 0.045           # Zero-lift drag coefficient [-]
    S_payload: float = 0.25      # Payload frontal area [m^2]
    Cd_payload: float = 1.05      # Payload drag coefficient [-]
    # Flight Conditions
    h_cruise: float = 100        # Cruise altitude [m]
    V_cruise: float = 20         # Cruise speed [m/s]
    # Mission
    R: float = 20000             # Range [m]
    # Materials
    foam_density: float = 48     # [kg/m^3]
    # Glass fibre sheet thickness used for skin mass calculations [mm]
    glass_sheet_thickness_mm: float = 1.0
    # Tail
    # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
    Vv: float = 0.04
    Vh: float = 0.50
    # Tail moment arm lh, wing AC → tail AC (the quantity that goes into the
    # tail-volume coefficients Vh = Sh·lh/(Sw·c) and Vv = Sv·lh/(Sw·b), and into
    # the stability/controllability scissor). The physical boom length L_boom
    # is derived from lh and the wing/aileron/tail geometry; it is strictly
    # longer than lh because the actual tail extends past its AC to its TE.
    # When None, run() falls back to deriving lh from the Vh tail-volume estimate.
    lh: float | None = None
    # Optional override for the horizontal tail area. When None, run() derives
    # it from the tail volume coefficient (or bh²/ARt if lh is also None).
    # The pipeline loop overwrites this with the scissor-plot result each pass.
    Sh: float | None = None
    # https://www.fmsg-alling.de/wp-content/uploads/2013/09/V-Leitwerke.pdf
    ARt: float = 4.5
    lam_t: float = 1.0


@dataclass
class SizingResult:
    inputs: SizingInputs
    t_over_c_root: float  # supplied externally (from the chosen airfoil)
    # Mass
    m_drone_loaded: float
    m_wing: float
    m_tail: float
    # Atmosphere
    T_isa: float
    p: float
    rho: float
    # Mission
    t_cruise: float
    # Wing geometry
    Sw: float
    c: float
    c_root: float
    # Aerodynamics
    q_cruise: float
    e: float
    CL: float
    Cl_airfoil: float
    k: float
    Cd: float
    L: float
    D: float
    LD_ratio: float
    # Tail
    bh: float
    Sh: float
    # lh = tail moment arm (wing AC → tail AC). Used by the stability /
    # controllability scissor and by the Vh, Vv tail-volume coefficients.
    lh: float
    # L_boom = physical tail-boom length (aileron hinge → tail TE). Used for
    # the tail-rod cantilever sizing, the tail boom drag wetted area, the tail
    # mass arm, and the side-view plot. Always strictly longer than lh.
    L_boom: float
    Sv: float
    bv: float
    St: float
    bt: float
    ct: float
    tt: float


def tail_chord(yb: float, St: float, bt: float, lam_t: float) -> float:
    return (2 * St) / ((1 + lam_t) * bt) * (1 - (1 - lam_t) * (2 * yb))


def run(
    inputs: SizingInputs | None = None,
    *,
    t_over_c_root: float = 0.12,
    c_aileron_to_c_wing: float = 0.3,
) -> SizingResult:
    """Run initial sizing.

    `t_over_c_root` is supplied externally (typically from the chosen
    airfoil's geometry) since it's a property of the section, not a
    free design knob.

    `c_aileron_to_c_wing` is the aileron-to-wing chord ratio; it pins the
    aileron hinge x/c, which is the inboard end of the tail boom. It must
    match the value used by aileron sizing, so the pipeline supplies it from
    the same config block (`CONTROL_SURFACE.c_aileron_to_c_wing`).
    """
    if inputs is None:
        inputs = SizingInputs()
    i = inputs

    # Mass
    m_drone_loaded = i.m_drone_empty + (i.m_payload / i.n_drones)

    # Atmosphere (ISA)
    T_isa = 15.00 + 273.15 - 0.0065 * i.h_cruise
    p = 101.325e3 * (T_isa / 288.15) ** 5.2559
    rho = p / (R_air * T_isa)

    # Mission profile
    t_cruise = i.R / i.V_cruise

    # Wing geometry
    Sw = i.b ** 2 / i.AR
    c = Sw / i.b
    c_root = 2 * Sw / (i.b * (1 + i.lam))

    # Aerodynamics
    q_cruise = 0.5 * rho * i.V_cruise ** 2
    e = 1.78 * (1 - 0.045 * i.AR ** 0.68) - 0.64
    CL = (m_drone_loaded * g0) / (q_cruise * Sw)
    Cl_airfoil = (i.AR + 2) * CL / i.AR
    k = 1 / (np.pi * e * i.AR)
    Cd = i.Cd0 + k * CL ** 2 + (i.Cd_payload * i.S_payload / (i.n_drones * Sw))
    L = CL * q_cruise * Sw
    D = Cd * q_cruise * Sw
    LD_ratio = L / D

    # Structural
    m_wing = i.foam_density * (t_over_c_root * c) * Sw

    # Tail geometry
    bh = np.sqrt(i.Sh * i.ARt) if i.Sh is not None else (i.b * 0.365445026178)  # [m] Desmos
    if i.Sh is not None:
        Sh = i.Sh
    elif i.lh is None:
        Sh = bh ** 2 / i.ARt
    else:
        Sh = i.Vh * Sw * c / i.lh
    lh = i.lh if i.lh is not None else (i.Vh * Sw * c / Sh)
    Sv = i.Vv * Sw * i.b / lh
    bv = np.sqrt(2 * i.ARt * Sv) / 2
    St = Sh + Sv
    bt = np.sqrt(bh ** 2 + bv ** 2)
    ct = tail_chord(0.5, St, bt, i.lam_t)
    tt = ct * t_over_c_root
    m_tail = i.foam_density * St * tt

    # Physical tail boom length, datum at LEMAC:
    #   L_boom = lh − (x_aileron_hinge − x_ac_wing) + 0.75·ct
    # i.e. boom spans from the aileron hinge (its inboard structural anchor)
    # to the tail TE (its outboard end). lh is wing AC → tail AC, so the
    # correction subtracts the wing AC → aileron-hinge offset and adds the
    # tail AC → tail TE offset (0.75·ct, AC at quarter chord).
    x_ac_wing = 0.25 * c
    x_aileron_hinge = (1.0 - c_aileron_to_c_wing) * c_root
    L_boom = lh - (x_aileron_hinge - x_ac_wing) + 0.75 * ct

    return SizingResult(
        inputs=inputs,
        t_over_c_root=t_over_c_root,
        m_drone_loaded=m_drone_loaded,
        m_wing=m_wing,
        m_tail=m_tail,
        T_isa=T_isa,
        p=p,
        rho=rho,
        t_cruise=t_cruise,
        Sw=Sw,
        c=c,
        c_root=c_root,
        q_cruise=q_cruise,
        e=e,
        CL=CL,
        Cl_airfoil=Cl_airfoil,
        k=k,
        Cd=Cd,
        L=L,
        D=D,
        LD_ratio=LD_ratio,
        bh=bh,
        Sh=Sh,
        lh=lh,
        L_boom=L_boom,
        Sv=Sv,
        bv=bv,
        St=St,
        bt=bt,
        ct=ct,
        tt=tt,
    )


def summary(r: SizingResult) -> None:
    print("\n--- Atmosphere ---")
    print(f"  ISA Temperature      : {r.T_isa:.2f}  °C")
    print(f"  Pressure             : {r.p:.1f}  Pa")
    print(f"  Air Density          : {r.rho:.4f}  kg/m³")

    print("\n--- Mass ---")
    print(f"  Loaded drone mass    : {r.m_drone_loaded:.2f}  kg")
    print(f"  Wing mass            : {r.m_wing:.3f}  kg")
    print(f"  Tail mass            : {r.m_tail:.3f}  kg")

    print("\n--- Wing Geometry ---")
    print(f"  Wing area            : {r.Sw:.4f}  m²")
    print(f"  Mean chord           : {r.c:.4f}  m")

    print("\n--- Aerodynamics ---")
    print(f"  Dynamic pressure     : {r.q_cruise:.2f}  Pa")
    print(f"  Oswald efficiency    : {r.e:.4f}")
    print(f"  CL (drone)           : {r.CL:.4f}")
    print(f"  Cl (airfoil)         : {r.Cl_airfoil:.4f}")
    print(f"  CD                   : {r.Cd:.5f}")
    print(f"  L/D ratio            : {r.LD_ratio:.2f}")
    print(f"  Lift                 : {r.L:.2f}  N")
    print(f"  Drag                 : {r.D:.2f}  N")

    print("\n--- Mission ---")
    print(f"  Cruise time          : {r.t_cruise:.1f}  s  ({r.t_cruise / 60:.1f} min)")

    print("\n--- Tail ---")
    print(f"  Horizontal tail area : {r.Sh:.4f}  m²")
    print(f"  Vertical tail area   : {r.Sv:.4f}  m²")
    print(f"  lh (AC → AC)         : {r.lh:.4f}  m")
    print(f"  L_boom (boom length) : {r.L_boom:.4f}  m")
    print(f"  Tail chord           : {r.ct:.4f}  m")


if __name__ == "__main__":
    summary(run(SizingInputs()))
