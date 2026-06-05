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
    gust_speed: float = 5.0      # Gust increment used for structures/control sizing [m/s]
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
    # Physical tail-boom length from aileron hinge to vertical-tail trailing edge.
    # run() derives lh (wing AC -> tail AC) from this and the tail geometry.
    L_boom: float = 1.50
    # Optional override for the horizontal tail area. When None, run() derives
    # it from the tail volume coefficient using the boom-derived lh.
    # The pipeline loop overwrites this with the scissor-plot result each pass.
    Sh: float | None = None
    # https://www.fmsg-alling.de/wp-content/uploads/2013/09/V-Leitwerke.pdf
    ARt: float = 4.5
    lam_t: float = 1.0
    # Static stability margin x_cg/c [-]. Typical range 0.05–0.15.
    # Larger values give more inherent stability but increase trim drag and
    # tail-load demand. Set in config.yaml under sizing: SM.
    SM: float = 0.05


@dataclass
class SizingResult:
    inputs: SizingInputs
    t_over_c_root: float  # supplied externally (from the chosen airfoil)
    # Mass
    m_drone_loaded: float
    m_wing: float
    m_tail_h: float
    m_tail_v: float
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
    V_structural: float
    q_structural: float
    e: float
    CL: float
    CL_one_drone_failure: float
    lift_one_drone_failure: float
    lift_one_drone_failure_gust: float
    lift_gust_increment: float
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
    ch: float
    tt_h: float
    # L_boom = physical tail-boom length (aileron hinge → tail TE). Used for
    # the tail-rod cantilever sizing, the tail boom drag wetted area, the tail
    # mass arm, and the side-view plot. Always strictly longer than lh.
    ARt : float
    L_boom: float
    Sv: float
    bv: float
    cv: float
    tt_v: float
    gamma_air: float
    R_air: float


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
    V_structural = i.V_cruise + i.gust_speed
    q_structural = 0.5 * rho * V_structural ** 2
    e = 1.78 * (1 - 0.045 * i.AR ** 0.68) - 0.64
    CL = (m_drone_loaded * g0) / (q_cruise * Sw)
    if i.n_drones <= 1:
        raise ValueError("n_drones must be greater than 1 for one-drone-failure sizing.")
    m_one_drone_failure = (i.n_drones * i.m_drone_empty + i.m_payload) / (i.n_drones - 1)
    CL_one_drone_failure = (m_one_drone_failure * g0) / (q_cruise * Sw)
    lift_one_drone_failure = CL_one_drone_failure * q_cruise * Sw
    lift_one_drone_failure_gust = CL_one_drone_failure * q_structural * Sw
    lift_gust_increment = lift_one_drone_failure_gust - lift_one_drone_failure
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
    ch = bh / i.ARt
    L_boom = i.L_boom
    x_aileron_hinge = (1.0 - c_aileron_to_c_wing) * c_root
    x_wing_ac = 0.25 * c
    x_tail_ac_from_boom_end = 0.75 * ch
    lh = L_boom + x_aileron_hinge - x_wing_ac - x_tail_ac_from_boom_end
    if lh <= 0.0:
        raise ValueError(
            f"Tail geometry failed: derived lh = {lh:.3f} m from "
            f"L_boom = {L_boom:.3f} m. Increase L_boom or reduce tail chord."
        )
    if i.Sh is not None:
        Sh = i.Sh
    else:
        Sh = i.Vh * Sw * c / lh
    Sv = i.Vv * Sw * i.b / lh
    bv = np.sqrt(2 * i.ARt * Sv) / 2
    cv = 2 * bv / i.ARt
    tt_h = ch * t_over_c_root
    tt_v = cv * t_over_c_root
    m_tail_h = i.foam_density * Sh * tt_h
    m_tail_v = i.foam_density * Sv * tt_v

    return SizingResult(
        inputs=inputs,
        t_over_c_root=t_over_c_root,
        m_drone_loaded=m_drone_loaded,
        m_wing=m_wing,
        m_tail_h=m_tail_h,
        m_tail_v=m_tail_v,
        T_isa=T_isa,
        p=p,
        rho=rho,
        t_cruise=t_cruise,
        Sw=Sw,
        c=c,
        c_root=c_root,
        q_cruise=q_cruise,
        V_structural=V_structural,
        q_structural=q_structural,
        e=e,
        CL=CL,
        CL_one_drone_failure=CL_one_drone_failure,
        lift_one_drone_failure=lift_one_drone_failure,
        lift_one_drone_failure_gust=lift_one_drone_failure_gust,
        lift_gust_increment=lift_gust_increment,
        Cl_airfoil=Cl_airfoil,
        k=k,
        Cd=Cd,
        L=L,
        D=D,
        LD_ratio=LD_ratio,
        bh=bh,
        Sh=Sh,
        lh=lh,
        ARt=i.ARt,
        L_boom=L_boom,
        Sv=Sv,
        bv=bv,
        tt_h=tt_h,
        tt_v=tt_v,
        ch=ch,
        cv=cv,
        gamma_air=gamma_air,
        R_air=R_air,
    )


def summary(r: SizingResult) -> None:
    print("\n--- Atmosphere ---")
    print(f"  ISA Temperature      : {r.T_isa:.2f}  °C")
    print(f"  Pressure             : {r.p:.1f}  Pa")
    print(f"  Air Density          : {r.rho:.4f}  kg/m³")

    print("\n--- Mass ---")
    print(f"  Loaded drone mass    : {r.m_drone_loaded:.2f}  kg")
    print(f"  Wing mass            : {r.m_wing:.3f}  kg")
    print(f"  Horizontal tail mass : {r.m_tail_h:.3f}  kg")
    print(f"  Vertical tail mass   : {r.m_tail_v:.3f}  kg")

    print("\n--- Wing Geometry ---")
    print(f"  Wing area            : {r.Sw:.4f}  m²")
    print(f"  Mean chord           : {r.c:.4f}  m")

    print("\n--- Aerodynamics ---")
    print(f"  Dynamic pressure     : {r.q_cruise:.2f}  Pa")
    print(f"  Gust speed           : {r.inputs.gust_speed:.2f}  m/s")
    print(f"  Structural speed     : {r.V_structural:.2f}  m/s")
    print(f"  Structural q         : {r.q_structural:.2f}  Pa")
    print(f"  Oswald efficiency    : {r.e:.4f}")
    print(f"  CL (drone)           : {r.CL:.4f}")
    print(f"  CL (1-drone failure) : {r.CL_one_drone_failure:.4f}")
    print(f"  Gust lift increment  : {r.lift_gust_increment:.2f}  N")
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
    print(f"  Horizontal tail chord: {r.ch:.4f}  m")
    print(f"  Vertical tail chord  : {r.cv:.4f}  m")
    print(f"  Horizontal tail span : {r.bh:.4f}  m")
    print(f"  Vertical tail span   : {r.bv:.4f}  m")

    print("\n--- Stability ---")
    print(f"  Static margin (SM)   : {r.inputs.SM:.3f}  (x_cg/c)")


if __name__ == "__main__":
    summary(run(SizingInputs()))