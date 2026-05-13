"""Design knobs for the sizing pipeline.

Edit values here and run `python main.py`. Every module-level constant in
this file is consumed by `pipeline.loop.run_pipeline`.
"""
from __future__ import annotations

import numpy as np

from propulsion.sizing import PropulsionInputs
from sizing.aileron import AileronInputs
from sizing.fuselage import FuselageInputs
from sizing.wing import SizingInputs
from structures.rods import RodInputs


SIZING = SizingInputs(
    m_payload=60,
    m_drone_empty=10,
    n_drones=4,
    b=3.0,
    AR=7.5,
    lam=1.0,
    Cd0=0.03,            # initial guess; refined after fuselage sizing
    S_payload=0.25,
    Cd_payload=1.0,
    h_cruise=100,
    V_cruise=20,
    R=20000,
    foam_density=48,
    Vv=0.04,
    Vh=0.50,
    ARt=4.5,
    lam_t=1.0,
)

PROPULSION = PropulsionInputs(
    csv_prop="data/20x10E_performance.csv",
    n_props=2,
    D_prop=20 * 0.0254,  # [m] — derived from csv_prop name if None
    prop_mass=0.1,       # [kg] — mass per propeller
    # Motor / drivetrain
    eff_motor=0.8,
    T_W_to=2.0,          # thrust-to-weight for take-off / VTOL [-]
    throttle_max=0.9,    # design throttle cap at the sizing RPM [-]
    # Climb segment
    climb_rate=10 / 3,   # [m/s]
    t_climb=30,          # [s]
    # Battery
    n_cells=6,
    voltage_cell=3.7,    # [V]
    battery_density=250, # [Wh/L]
    # Propeller solver tuning (shared between cruise & climb)
    cruise_rpm_init=7000,
    climb_rpm_init=12000,
    rpm_tol=1.0,
    thrust_tol=1.0,
    max_iter=200,
    rpm_step=50.0,
    # VTOL/hover solver tuning — looser thrust tol because the table-step
    # resolution (rpm_step) easily exceeds 0.5 N near hover RPM.
    vtol_rpm_init=5000,
    vtol_thrust_tol=2.0,
    vtol_max_iter=400,
    vtol_rpm_step=25.0,
    # Verbosity — solver prints per-iteration tables; off by default so the
    # main.py iteration loop stays readable.
    verbose=False,
)

# Battery dimensions: off-the-shelf envelope unless both aspect ratios are > 0,
# in which case dimensions are recomputed from the required battery volume.
# AR_lw / AR_lh below match the off-the-shelf 0.212 × 0.090 × 0.060 cell so the
# starting shape is preserved but the cell scales with the propulsion-sized volume.
FUSELAGE = FuselageInputs(
    battery_length=0.212,
    battery_width=0.090,
    battery_height=0.060,
    AR_lw=0.212 / 0.090,  # length / width  (≈ 2.356)
    AR_lh=0.212 / 0.060,  # length / height (≈ 3.533)
    housing_factor=1.5,
    casing_factor=1.1,
)

CONTROL_SURFACE = AileronInputs(
    cl_alpha=2 * np.pi,
    cd0_section=0.04,
    roll_req_deg=60.0,
    max_da_deg=10.0,
    max_y_frac=0.8,
    c_aileron_to_c_wing=0.3,
)

# CFRP rod sizing — both rods solved closed-form for the thinnest wall that
# passes both a tip-deflection (defl_max) and a compressive-stress
# (s_c / safety_factor) limit. d is fixed by geometry (rod must fit inside
# the local section thickness). tail_tc must match TAIL_AIRFOIL.
STRUCTURE = RodInputs(
    safety_factor=1.2,
    defl_max=0.05,             # [m] 50 mm tip deflection budget
    d_to_section_ratio=0.8,    # rod OD as fraction of section thickness
    CLt_max=1.0,
    tail_tc=0.10,              # matches NACA0010 tail
)

# Wing airfoil: .dat path or NACA digits (e.g. "2412"). If None, the pipeline
# prompts at runtime.
AIRFOIL: str | None = "airfoils/NACA4412.dat"

# Tail airfoil: .dat path or NACA digits. Symmetric sections are typical.
TAIL_AIRFOIL: str = "airfoils/NACA0010.dat"

# Alpha sweep used to build the wing drag polar for the CL/CD plot [deg].
ALPHA_SWEEP_DEG = (-2.0, 12.0, 30)
# Coarser sweep used inside the iteration loop to find max-L/D CL each pass.
ALPHA_SWEEP_LOOP_DEG = (-2.0, 12.0, 15)

# Sizing↔propulsion↔fuselage↔drag↔mass↔wing-area iteration.
#   MASS_CLOSURE = False  → hold m_drone_empty at SIZING.m_drone_empty (requirement).
#   MASS_CLOSURE = True   → feed total drone mass back into m_drone_empty each pass.
#   SW_CLOSURE   = False  → hold b, AR (and therefore Sw) at the SIZING values.
#   SW_CLOSURE   = True   → resize Sw each pass so the operating CL sits at the max
#                           L/D point of the drone+payload polar; AR is held fixed,
#                           so b = sqrt(AR · Sw) floats.
# Convergence requires every active criterion (CD0 always; m_drone if MASS_CLOSURE;
# Sw if SW_CLOSURE) to be inside its tolerance.
MASS_CLOSURE = True
SW_CLOSURE = True
N_ITER_MAX = 20
CD0_TOL = 1e-4
MASS_TOL = 1e-3   # kg
SW_TOL = 1e-3     # m²
