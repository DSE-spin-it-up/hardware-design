"""Pipeline plots: convergence history, drone L/D polars, and CG side view."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from aerodynamics.airfoil_geometry import AirfoilGeometry
from sizing.fuselage import FuselageResult
from sizing.rudder import RudderResult
from sizing.wing import SizingResult
from structures.rods import RodResult
from propulsion.sizing import PropulsionResult
from weights.mass import compute_y_cg, _tube_y_bounds


def plot_convergence(
    CD0_history: list[float],
    mass_history: list[float],
    Sw_history: list[float],
    cg_history: list[float],
    Cd0_guess: float,
    m_drone_guess: float,
    Sw_guess: float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    cd_iters = np.arange(len(CD0_history) + 1)
    cd_values = np.array([Cd0_guess, *CD0_history])
    axes[0].plot(cd_iters, cd_values, "o-")
    axes[0].axhline(cd_values[-1], color="g", ls="--",
                    label=f"converged CD0 = {cd_values[-1]:.5f}")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("CD0")
    axes[0].set_title("CD0 progression")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    m_iters = np.arange(len(mass_history) + 1)
    m_values = np.array([m_drone_guess, *mass_history])
    axes[1].plot(m_iters, m_values, "o-")
    axes[1].axhline(m_values[-1], color="g", ls="--",
                    label=f"converged m_drone = {m_values[-1]:.3f} kg")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("drone mass [kg]")
    axes[1].set_title("Drone mass progression")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    sw_iters = np.arange(len(Sw_history) + 1)
    sw_values = np.array([Sw_guess, *Sw_history])
    axes[2].plot(sw_iters, sw_values, "o-")
    axes[2].axhline(sw_values[-1], color="g", ls="--",
                    label=f"converged Sw = {sw_values[-1]:.4f} m²")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("wing area Sw [m²]")
    axes[2].set_title("Wing area progression")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # CG only exists from iter 1 onward (depends on propulsion result).
    cg_iters = np.arange(1, len(cg_history) + 1)
    cg_values = np.array(cg_history)
    axes[3].plot(cg_iters, cg_values, "o-")
    axes[3].axhline(cg_values[-1], color="g", ls="--",
                    label=f"converged x_cg = {cg_values[-1]:.4f} m")
    cg_excursion = cg_values.max() - cg_values.min()
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("x_cg from LEMAC [m]")
    axes[3].set_title(f"CG excursion (Δ = {cg_excursion * 1000:.1f} mm)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    plt.show()


def plot_drone_ld(
    CL: np.ndarray,
    CD_drone: np.ndarray,
    CD_full: np.ndarray,
    CL_op: float,
    airfoil_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, CD, title in [
        (axes[0], CD_drone, "Drone only"),
        (axes[1], CD_full, "Drone + payload"),
    ]:
        LD = CL / CD
        idx_max = int(np.argmax(LD))
        ax.plot(CL, LD, "o-", label="L/D")
        ax.axvline(CL_op, color="r", ls="--", label=f"operating CL = {CL_op:.3f}")
        ax.plot(
            CL[idx_max], LD[idx_max], "g*", ms=14,
            label=f"max L/D = {LD[idx_max]:.1f} @ CL = {CL[idx_max]:.2f}",
        )
        ax.set_xlabel("CL")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("L/D")
    fig.suptitle(f"Drag polar — airfoil {airfoil_name}")
    fig.tight_layout()
    plt.show()


def plot_cg_side_view(
    sizing: SizingResult,
    fus: FuselageResult,
    struct: RodResult,
    cg: dict[str, float],
    masses: dict[str, float],
    airfoil_path: str | Path,
    tail_airfoil_path: str | Path | None = None,
    propulsion: PropulsionResult | None = None,
    rudder: RudderResult | None = None,
    y_cg: dict[str, float] | None = None,
    show: bool = True,
) -> plt.Figure:
    """Side-view sketch of the aircraft with component CGs and overall CG.

    Fuselage is drawn as a Raymer-style body:
      - ogive nose cone  (triangle)
      - parallel cylinder = structural box  (rectangle)
      - tail taper  (triangle)
    The dashed internal structural box is overlaid for reference.
    Sensor payload and wiring harness CG markers are also shown when present.
    """
    # ------------------------------------------------------------------ airfoil
    airfoil = AirfoilGeometry(airfoil_path)
    root_chord = sizing.c_root
    airfoil_coords = np.column_stack((
        airfoil.polygon[:, 0] * root_chord,
        airfoil.polygon[:, 1] * root_chord,
    ))
    airfoil_y_offset = -float(np.min(airfoil_coords[:, 1]))
    airfoil_coords[:, 1] += airfoil_y_offset

    # ------------------------------------------------------------------ CG x-positions
    x_motor       = cg["motors"]
    x_motor_back  = cg.get("motor_back")
    x_rod_wing    = cg["rod_spar"]
    x_rod_aileron = cg["rod_aileron"]
    x_batt        = cg["battery"]
    x_tail        = cg["tail"]
    x_overall     = cg["overall"]

    prop_radius = propulsion.D_prop / 2.0 if propulsion is not None else 0.0

    # ------------------------------------------------------------------ tail geometry
    c_v      = sizing.cv
    ch       = sizing.ch
    L_boom   = sizing.L_boom
    tail_start = x_batt - 0.5 * fus.battery_length
    x_vt_te  = tail_start + L_boom
    x_vt_le  = x_vt_te - c_v
    x_vt_fs  = x_vt_le + 0.25 * c_v
    if x_motor_back is None:
        x_motor_back = x_vt_fs
    x_ht_te  = x_vt_fs - prop_radius
    x_ht_le  = x_ht_te - ch
    x_tail_root_v = x_vt_le
    x_tail_root_h = x_ht_le
    x_tail_end    = x_vt_te
    b_v_total     = sizing.bv

    # ------------------------------------------------------------------ tail airfoil
    tail_airfoil_coords = None
    x_ht_spar = None
    x_ht_control = None
    x_vt_hinge = None
    rudder_chord = None
    rudder_span = None
    if tail_airfoil_path is not None:
        tail_airfoil = AirfoilGeometry(tail_airfoil_path)
        tail_airfoil_coords = np.column_stack((
            tail_airfoil.polygon[:, 0] * sizing.ch,
            tail_airfoil.polygon[:, 1] * sizing.ch,
        ))
        _, x_mt = tail_airfoil.compute_maximum_thickness()
        x_ht_spar    = x_tail_root_h + x_mt * sizing.ch
        x_ht_control = x_tail_root_h + (1.0 - struct.inputs.c_ruddervator_to_c_tail) * sizing.ch
    if rudder is not None:
        x_vt_hinge   = x_tail_root_v + (1.0 - rudder.geometry.cR_cV) * c_v
        rudder_chord  = rudder.geometry.cR_cV * c_v
        rudder_span   = rudder.geometry.bR_bV * b_v_total
    elif tail_airfoil_path is not None:
        x_vt_hinge = x_tail_root_v + (1.0 - struct.inputs.c_ruddervator_to_c_tail) * c_v

    # ------------------------------------------------------------------ rod y-positions
    _, y_up_s, y_lo_s = airfoil.compute_thickness(x_rod_wing    / root_chord)
    _, y_up_a, y_lo_a = airfoil.compute_thickness(x_rod_aileron / root_chord)
    y_rod_spar    = 0.5 * (y_up_s + y_lo_s) * root_chord + airfoil_y_offset
    y_rod_aileron = 0.5 * (y_up_a + y_lo_a) * root_chord + airfoil_y_offset
    y_ht_spar     = y_rod_aileron
    y_ht_control  = y_rod_aileron
    if tail_airfoil_path is not None and x_ht_spar is not None:
        xn = (x_ht_spar - x_tail_root_h) / sizing.ch
        _, y_up_ht_s, y_lo_ht_s = tail_airfoil.compute_thickness(xn)
        y_ht_spar = y_rod_aileron + 0.5 * (y_up_ht_s + y_lo_ht_s) * sizing.ch
    if tail_airfoil_path is not None and x_ht_control is not None:
        xn = (x_ht_control - x_tail_root_h) / sizing.ch
        _, y_up_ht_c, y_lo_ht_c = tail_airfoil.compute_thickness(xn)
        y_ht_control = y_rod_aileron + 0.5 * (y_up_ht_c + y_lo_ht_c) * sizing.ch

    rod_radius_w = struct.d_w       / 2.0
    rod_radius_a = struct.d_aileron / 2.0

    # ------------------------------------------------------------------ tube / battery
    tube_x0     = x_rod_wing - fus.inputs.tube_tail_overlap
    tube_x1     = x_rod_aileron + fus.inputs.tube_tail_overlap
    tube_length = tube_x1 - tube_x0
    tube_y0, tube_y1 = _tube_y_bounds(struct, fus, y_rod_spar, y_rod_aileron)
    tube_height = tube_y1 - tube_y0
    tube_yc     = tube_y0 + tube_height / 2.0
    lowest_wing_rod_bottom = min(
        y_rod_spar - struct.d_spar / 2.0,
        y_rod_aileron - struct.d_aileron / 2.0,
    )
    wing_y_shift = tube_y1 - lowest_wing_rod_bottom
    y_tail_base = tube_yc
    airfoil_coords[:, 1] += wing_y_shift
    y_rod_spar += wing_y_shift
    y_rod_aileron += wing_y_shift
    y_ht_spar = y_tail_base
    y_ht_control = y_tail_base

    battery_length = fus.battery_length
    battery_height = fus.battery_height
    if y_cg is None:
        y_cg = compute_y_cg(sizing, fus, struct, masses, airfoil_path, cg=cg)
    batt_x0 = x_batt - battery_length / 2.0
    batt_y0 = y_cg.get("battery_bottom", y_cg["battery"] - battery_height / 2.0)

    # ------------------------------------------------------------------ structural box vertical placement
    box_y0 = 0.0
    box_yc = box_y0 + fus.box_height / 2.0
    body_y0 = 0.0
    body_yc = body_y0 + fus.height / 2.0

    # ------------------------------------------------------------------ Raymer fuselage x-coordinates
    box_x0      = fus.x_nose + fus.l_nose
    box_x1      = box_x0 + fus.l_cylinder
    fus_nose_x  = fus.x_nose
    fus_tail_x  = box_x1 + fus.l_tail

    half_h = fus.height / 2.0

    # Nose cone: triangle  tip → top-left → bottom-left
    nose_poly = np.array([
        [fus_nose_x, body_yc],
        [box_x0,     body_yc + half_h],
        [box_x0,     body_yc - half_h],
    ])
    tail_exit_radius = struct.d_t / 2.0
    # Tail cone: top/bottom of cylinder taper to the tail-boom outer diameter.
    tail_poly = np.array([
        [box_x1,     body_yc + half_h],
        [box_x1,     body_yc - half_h],
        [fus_tail_x, y_tail_base - tail_exit_radius],
        [fus_tail_x, y_tail_base + tail_exit_radius],
    ])

    plot_height = max(
        body_yc + half_h + 0.02,
        box_yc + fus.box_height / 2.0 + 0.02,
        batt_y0 + battery_height + 0.01,
        y_tail_base + b_v_total + 0.05,
    )

    # ================================================================== draw
    fig, ax = plt.subplots(figsize=(16, 5))

    # --- Raymer fuselage body ---
    fus_kw = dict(facecolor="lightcyan", edgecolor="navy", alpha=0.5, linewidth=2)
    ax.add_patch(Rectangle((box_x0, body_y0), fus.l_cylinder, fus.height,
                            label="Fuselage cylinder", **fus_kw))
    ax.add_patch(Polygon(nose_poly, closed=True, label="Fuselage nose", **fus_kw))
    ax.add_patch(Polygon(tail_poly, closed=True, label="Fuselage tail", **fus_kw))

    # --- Internal structural box (dashed) ---
    ax.add_patch(Rectangle((box_x0, box_y0), fus.box_length, fus.box_height,
                            fill=False, linestyle="--", edgecolor="gray",
                            linewidth=1.2, label="Structural box"))

    # --- Wing airfoil ---
    ax.add_patch(Polygon(airfoil_coords, closed=True,
                         facecolor="lightblue", edgecolor="navy", alpha=0.6,
                         label="Wing profile"))

    # --- Battery ---
    ax.add_patch(Rectangle((batt_x0, batt_y0), battery_length, battery_height,
                            color="orange", alpha=0.5, label="Battery"))

    # --- Vertical tail ---
    ax.add_patch(Rectangle((x_tail_root_v, y_tail_base), c_v, b_v_total,
                            facecolor="lightsteelblue", edgecolor="navy",
                            alpha=0.2, label="Vertical tail"))
    if x_vt_hinge is not None:
        rw = rudder_chord if rudder_chord is not None else (x_tail_root_v + c_v - x_vt_hinge)
        rh = rudder_span  if rudder_span  is not None else b_v_total
        ax.add_patch(Rectangle((x_vt_hinge, y_tail_base), rw, rh,
                                facecolor="lightcoral", alpha=0.2, label="Rudder surface"))
        ax.plot([x_vt_hinge, x_vt_hinge], [y_tail_base, y_tail_base + rh],
                color="darkred", linewidth=2, linestyle="--", label="Rudder hinge")

    vt_rod_y0    = y_tail_base
    vt_spar_w    = struct.d_spar_vt
    vt_rud_w     = struct.d_control_vt
    ax.add_patch(Rectangle(
        (x_tail_root_v + c_v * 0.25 - vt_spar_w / 2.0, vt_rod_y0),
        vt_spar_w, b_v_total,
        facecolor="purple", alpha=0.7, label="VT spar rod",
    ))
    if x_vt_hinge is not None:
        ax.add_patch(Rectangle(
            (x_vt_hinge - vt_rud_w / 2.0, vt_rod_y0),
            vt_rud_w, b_v_total,
            facecolor="magenta", alpha=0.7, label="VT rudder rod",
        ))

    # --- Horizontal tail ---
    if tail_airfoil_coords is not None:
        tc = tail_airfoil_coords.copy()
        y_mid = 0.5 * (float(np.max(tc[:, 1])) + float(np.min(tc[:, 1])))
        tc[:, 0] += x_tail_root_h
        tc[:, 1] += y_tail_base - y_mid
        ax.add_patch(Polygon(tc, closed=True,
                             facecolor="lightgray", edgecolor="darkslategray",
                             alpha=0.6, label="Tail airfoil"))
    else:
        approx_t = max(0.05 * sizing.ch, 0.02)
        ax.add_patch(Rectangle(
            (x_tail_root_h, y_tail_base - approx_t / 2.0),
            sizing.ch, approx_t,
            facecolor="lightgray", edgecolor="darkslategray", alpha=0.6,
            label="Tail airfoil",
        ))

    if x_ht_spar is not None and x_ht_control is not None:
        ax.add_patch(Circle(
            (x_ht_spar, y_ht_spar),
            struct.d_spar_ht * 0.5,
            color="indigo",
            alpha=0.8,
            label="HT spar rod",
        ))
        ax.add_patch(Circle(
            (x_ht_control, y_ht_control),
            struct.d_control_ht * 0.5,
            color="darkorange",
            alpha=0.8,
            label="HT elevator rod",
        ))

    # --- Wing rods ---
    ax.add_patch(Circle((x_rod_wing,    y_rod_spar),    rod_radius_w, color="brown",  alpha=0.8, label="Wing rod"))
    ax.add_patch(Circle((x_rod_aileron, y_rod_aileron), rod_radius_a, color="sienna", alpha=0.8, label="Aileron rod"))

    # --- Tail boom ---
    tail_boom_d = struct.d_t
    ax.add_patch(Rectangle(
        (tail_start, y_tail_base - tail_boom_d / 2.0),
        x_tail_end - tail_start,
        tail_boom_d,
        facecolor="gray",
        edgecolor="dimgray",
        alpha=0.75,
        label="Tail boom",
    ))

    # --- Rod connectors ---
    connector_x = [
        cg.get("rod_connector_spar"),
        cg.get("rod_connector_aileron"),
        cg.get("rod_connector_payload"),
    ]
    connector_y = [
        y_cg.get("rod_connector_spar"),
        y_cg.get("rod_connector_aileron"),
        y_cg.get("rod_connector_payload"),
    ]
    connector_points = [
        (x, y) for x, y in zip(connector_x, connector_y)
        if x is not None and y is not None
    ]
    if connector_points:
        xs, ys = zip(*connector_points)
        ax.scatter(xs, ys, color="black", marker="s", s=55, zorder=8, label="Rod connector")

    # --- Servos (small dots behind rods so they don't obscure rod geometry) ---
    motor_y   = y_cg["motors"]
    overall_y = y_cg["overall"]

    x_servo_front   = cg["motors"]
    x_servo_aileron = cg["rod_aileron"]
    x_servo_rear    = x_motor_back
    x_servo_ht      = cg.get("ht_rud", cg["tail"])
    x_servo_vt      = cg.get("vt_rud", cg["tail"])
    ax.scatter(
        [x_servo_front,   x_servo_front,
         x_servo_aileron, x_servo_aileron,
         x_servo_rear,
         x_servo_ht,      x_servo_ht,
         x_servo_vt],
        [y_cg["servo_front"],   y_cg["servo_front"],
         y_cg["servo_aileron"], y_cg["servo_aileron"],
         y_cg["servo_rear"],
         y_cg["servo_ht"],      y_cg["servo_ht"],
         y_cg["servo_vt"]],
        color="teal", zorder=2, s=15, marker=".", label="Servo",
    )

    # --- CG markers ---
    ax.scatter([x_motor, x_overall], [motor_y, overall_y],
               color=["red", "black"], zorder=5)
    ax.text(x_motor,   motor_y   + 0.03, "Motor (front)", color="red",   ha="center", fontsize=8)
    ax.text(x_overall, overall_y + 0.02, "Overall CG",    color="black", ha="center", fontsize=8)
    ax.scatter([x_motor_back], [y_tail_base + b_v_total],
               color="red", zorder=5, s=100, marker="^")
    ax.text(x_motor_back, y_tail_base + b_v_total + 0.03, "Motor (rear)",
            color="red", ha="center", fontsize=8)

    # --- Sensor payload CG ---
    x_sensor = cg.get("sensors")
    y_sensor = y_cg.get("sensors")
    if x_sensor is not None and y_sensor is not None:
        ax.scatter(x_sensor, y_sensor, color="gold", marker="*", s=250,
                   edgecolors="black", zorder=7, label="Sensor Payload")

    # --- Wiring harness CG centres ---
    x_wire_w = cg.get("wiring_wing")
    x_wire_t = cg.get("wiring_tail")
    x_wire_s = cg.get("wiring_signal")
    if x_wire_w is not None and x_wire_t is not None and x_wire_s is not None:
        y_wire_w = y_cg.get("wiring_wing", tube_yc)
        y_wire_t = y_cg.get("wiring_tail", tube_yc)
        y_wire_s = y_cg.get("wiring_signal", tube_yc)
        ax.scatter(
            [x_wire_w, x_wire_t, x_wire_s],
            [y_wire_w, y_wire_t, y_wire_s],
            color="cyan", marker="X", s=80, edgecolors="black",
            zorder=6, label="Wiring Harness Centers",
        )

    # ------------------------------------------------------------------ axes
    x_left  = min(fus_nose_x - 0.05, -0.05)
    x_right = max(fus_tail_x, x_tail, x_overall) + 0.2
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(min(-0.05, box_y0 - 0.05), plot_height + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Aircraft CG side view")
    ax.set_xlabel("x [m] from LEMAC")
    ax.set_ylabel("vertical position [m]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.2)
    fig.subplots_adjust(right=0.76)

    if show:
        plt.show()
    return fig


def plot_scissor(scissor, *, show: bool = True):
    """Scissor plot: required S_h/S vs x_cg from LEMAC (stability + controllability).

    Green region above BOTH lines is the feasible design space — for any x_cg,
    the actual S_h/S must satisfy both constraints. The orange ⭕ marks the
    current operating point (converged x_cg, current S_h/S).
    """
    s = scissor
    fig, ax = plt.subplots(figsize=(9, 5))

    stab_pos = np.clip(s.ShS_stab, 0.0, None)
    ctrl_pos = np.clip(s.ShS_ctrl, 0.0, None)
    envelope = np.maximum(stab_pos, ctrl_pos)
    y_top = max(
        s.ShS_current * 1.6,
        float(np.nanmax(envelope)) * 1.15,
        0.05,
    )

    ax.plot(s.x_cg, s.ShS_stab, color="C0", lw=2.0, label="Stability boundary")
    ax.plot(s.x_cg, s.ShS_ctrl, color="C3", lw=2.0, label="Controllability boundary")
    ax.fill_between(s.x_cg, envelope, y_top,
                    color="green", alpha=0.12, label="Feasible")
    ax.fill_between(s.x_cg, 0.0, envelope,
                    color="red", alpha=0.08, label="Infeasible")

    ax.axhline(s.ShS_current, color="k", ls="--", lw=1.2,
               label=f"current $S_h/S$ = {s.ShS_current:.3f}")
    ax.axvline(s.x_cg_current, color="orange", ls="--", lw=1.2,
               label=f"current $x_{{cg}}$ = {s.x_cg_current:.4f} m "
                     f"({s.x_cg_current / s.c:.2%} $\\bar c$)")
    ax.plot([s.x_cg_current], [s.ShS_current], "o", color="orange",
            mec="k", ms=10, zorder=5)

    ax2 = ax.twiny()
    ax2.set_xlim(s.x_cg[0] / s.c, s.x_cg[-1] / s.c)
    ax2.set_xlabel(r"$x_{cg} / \bar{c}$  [-]")

    ax.set_xlim(s.x_cg[0], s.x_cg[-1])
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel("$x_{cg}$ from LEMAC  [m]")
    ax.set_ylabel("$S_h / S$  [-]")
    ax.set_title(
        f"Scissor — SM={s.SM:.2f}, "
        f"$d\\varepsilon/d\\alpha$={s.dep_da:.3f}, "
        f"$V_h/V$={s.Vh_V:.2f},  "
        f"$C_{{L_h}}$={s.CL_h:+.3f},  $C_{{L_{{A-h}}}}$={s.CL_A_h:.3f},  "
        f"$C_{{m_{{ac}}}}$={s.Cm_ac:+.4f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
