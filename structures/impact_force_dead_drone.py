import math

def calculate_dynamic_snap_force(
    m_dead: float,
    m_swarm_empty: float,
    m_payload: float,
    v_cruise: float,
    cd: float,
    s_ref: float,
    l_rope: float,
    elongation_pct: float
) -> dict:
    """
    Calculates tether snap force taking into account the continuous decrease
    in aerodynamic drag as the failed drone loses velocity.
    """
    rho = 1.22  # Air density [kg/m^3]
    g = 9.81    # Gravity [m/s^2]
    
    # 1. System Constants
    m_swarm_total = m_swarm_empty + m_payload
    mu = (m_dead * m_swarm_total) / (m_dead + m_swarm_total)
    
    # UIAA Rope Stiffness
    uiaa_test_force = 80.0 * g
    stiffness_k = (uiaa_test_force / (elongation_pct / 100.0)) / l_rope

    # 2. Numerical Time Integration Loop (Euler Method)
    t = 0.0
    dt = 0.001  # 1 millisecond steps for high accuracy
    separation = 0.0
    v_gap = 0.0  # Initial velocity gap is 0
    
    gamma = (rho * s_ref * cd) / (2.0 * m_dead)

    while separation < l_rope:
        # Calculate absolute velocity of the dead drone through the air
        v_abs_dead = v_cruise - v_gap
        
        # Instantaneous drag and deceleration
        instantaneous_drag = 0.5 * rho * s_ref * cd * (v_abs_dead ** 2)
        deceleration = instantaneous_drag / m_dead
        
        # Step forward in time
        v_gap += deceleration * dt
        separation += v_gap * dt
        t += dt
        
        # Safety cutoff if it somehow never snaps
        if t > 20.0:
            break

    # 3. Peak Impact Force using the true velocity gap at the moment of snap
    peak_force_newtons = v_gap * math.sqrt(mu * stiffness_k)
    peak_force_kg = peak_force_newtons / g

    return {
        "time_to_snap_s": t,
        "delta_v_at_snap_m_s": v_gap,
        "final_drag_at_snap_N": 0.5 * rho * s_ref * cd * ((v_cruise - v_gap) ** 2),
        "peak_force_N": peak_force_newtons,
        "peak_force_kg": peak_force_kg,
        "reduced_mass_kg": mu,
        "stiffness_k_N_m": stiffness_k
    }

if __name__ == "__main__":
    inputs = {
        "m_dead": 10,
        "m_swarm_empty": 13.51,
        "m_payload": 50.0,
        "v_cruise": 20.0,
        "cd": 0.042,
        "s_ref": 1.38,
        "l_rope": 12.5,
        "elongation_pct": 9.5
    }

    results = calculate_dynamic_snap_force(**inputs)

    print("\n--- Upgraded Dynamic Drag Snap Analysis ---")
    print(f"Time to Snap (Variable Drag) : {results['time_to_snap_s']:.3f} s")
    print(f"True Velocity Gap at Snap    : {results['delta_v_at_snap_m_s']:.2f} m/s")
    print(f"Remaining Drag Force at Snap : {results['final_drag_at_snap_N']:.2f} N")
    print("-" * 43)
    print(f"DYNAMIC PEAK SNAP FORCE      : {results['peak_force_N']:.1f} N")
    print(f"STATIC EQUIVALENT            : {results['peak_force_kg']:.1f} kg")
    print("-------------------------------------------\n")