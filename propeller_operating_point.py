"""
Propeller Operating Point Finder
=================================
Finds the RPM (and corresponding J) at which the propeller produces
a target thrust at a given cruise speed.

Algorithm (faithful to the described approach):
  1. Assume an initial RPM.
  2. From RPM and cruise speed, J is fully determined: J = V / (n*D).
  3. Look up thrust at that J (interpolating the table at that RPM).
  4. If thrust is too low -> reduce RPM (lowers J, raises thrust).
     If thrust is too high -> increase RPM (raises J, lowers thrust).
  5. Repeat until thrust matches target within tolerance.

Note: the "start at peak-efficiency J" step translates to choosing an
initial RPM that puts J near peak efficiency. The user-supplied RPM_INIT
serves this role. The walk then adjusts RPM (and hence J) until thrust
converges - which is equivalent to the described iteration.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------
# USER INPUTS - edit these
# ---------------------------------------------
CSV_FILE      = "15x135-3_performance.csv"
TARGET_THRUST = 20.0           # [N]
CRUISE_SPEED  = 20.0           # [m/s]
DIAMETER      = 15 * 0.0254    # 15-inch prop -> metres  (1 in = 0.0254 m)

RPM_INIT      = 6000           # initial RPM guess
RPM_TOL       = 1              # [RPM]  convergence tolerance
THRUST_TOL    = 0.5            # [N]    convergence tolerance (matched to RPM_STEP)
MAX_ITER      = 200            # max iterations
RPM_STEP      = 50             # [RPM]  walk step size
# ---------------------------------------------


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["RPM", "J", "Efficiency", "Ct", "Thrust (N)"])
    df = df[df["Thrust (N)"] > 0]
    return df


def j_from_rpm(rpm, velocity, diameter):
    """Advance ratio from RPM, cruise speed, and diameter."""
    n_rps = rpm / 60.0
    return velocity / (n_rps * diameter)


def interp_thrust_and_eff(df, rpm, J):
    """
    Interpolate thrust (and efficiency) at a given RPM and J by
    linearly interpolating between the two nearest RPM datasets.
    Returns (thrust, efficiency), or (NaN, NaN) if J is out of range.
    """
    rpms = np.sort(df["RPM"].unique())
    rpm  = float(np.clip(rpm, rpms[0], rpms[-1]))

    idx_hi = int(np.searchsorted(rpms, rpm))
    idx_hi = max(1, min(idx_hi, len(rpms) - 1))
    idx_lo = idx_hi - 1

    rpm_lo, rpm_hi = rpms[idx_lo], rpms[idx_hi]
    w = 0.0 if rpm_lo == rpm_hi else (rpm - rpm_lo) / (rpm_hi - rpm_lo)

    def lookup(r):
        sub = df[df["RPM"] == r].sort_values("J")
        j_arr = sub["J"].values
        t_arr = sub["Thrust (N)"].values
        e_arr  = sub["Efficiency"].values
        cp_arr = sub["Cp"].values
        if J < j_arr[0] or J > j_arr[-1]:
            return np.nan, np.nan, np.nan
        return (float(np.interp(J, j_arr, t_arr)),
                float(np.interp(J, j_arr, e_arr)),
                float(np.interp(J, j_arr, cp_arr)))

    t_lo, e_lo, cp_lo = lookup(rpm_lo)
    t_hi, e_hi, cp_hi = lookup(rpm_hi)

    if np.isnan(t_lo) or np.isnan(t_hi):
        return np.nan, np.nan, np.nan

    return ((1 - w) * t_lo  + w * t_hi,
            (1 - w) * e_lo  + w * e_hi,
            (1 - w) * cp_lo + w * cp_hi)


def best_efficiency_rpm(df, velocity, diameter):
    """
    Scan all RPMs in the data and find the one where the operating J
    (set by cruise speed) falls closest to peak efficiency.
    Returns the RPM and its peak-efficiency J.
    """
    rpms = np.sort(df["RPM"].unique())
    best_rpm, best_eff, best_J = None, -np.inf, None

    for rpm in rpms:
        J   = j_from_rpm(rpm, velocity, diameter)
        sub = df[df["RPM"] == rpm].sort_values("J")
        j_arr = sub["J"].values
        e_arr = sub["Efficiency"].values
        if J < j_arr[0] or J > j_arr[-1]:
            continue
        eff = float(np.interp(J, j_arr, e_arr))
        if eff > best_eff:
            best_eff, best_rpm, best_J = eff, rpm, J

    return best_rpm, best_J


def solve(csv_file, target_thrust, cruise_speed, diameter,
          rpm_init, rpm_tol, thrust_tol, max_iter, rpm_step):

    df = load_data(csv_file)

    # --- Step 1: start from peak-efficiency RPM as described ---
    rpm_eff, j_eff = best_efficiency_rpm(df, cruise_speed, diameter)
    print(f"\n{'='*60}")
    print(f"  Target thrust  : {target_thrust:.2f} N")
    print(f"  Cruise speed   : {cruise_speed:.2f} m/s")
    print(f"  Diameter       : {diameter/0.0254:.1f} in  ({diameter*1000:.0f} mm)")
    print(f"  Peak-eff RPM   : {rpm_eff:.0f}  (J = {j_eff:.4f})")
    print(f"  Starting RPM   : {rpm_init:.0f}  (user override)")
    print(f"{'='*60}")
    print(f"{'Iter':>4}  {'RPM':>10}  {'J':>8}  {'Thrust':>8}  {'Error':>8}")
    print(f"{'----':>4}  {'----------':>10}  {'--------':>8}  {'--------':>8}  {'--------':>8}")

    rpm = float(rpm_init)

    # Determine initial walk direction: eval thrust at starting RPM
    J0     = j_from_rpm(rpm, cruise_speed, diameter)
    T0, _, _ = interp_thrust_and_eff(df, rpm, J0)
    if np.isnan(T0):
        print(f"\n  Cannot evaluate thrust at initial RPM {rpm:.0f} / J {J0:.4f}.")
        return None

    direction = -1 if T0 > target_thrust else +1   # lower RPM -> higher thrust

    for i in range(1, max_iter + 1):
        J = j_from_rpm(rpm, cruise_speed, diameter)
        T, eff, Cp = interp_thrust_and_eff(df, rpm, J)

        err = T - target_thrust if not np.isnan(T) else np.nan
        print(f"{i:>4}  {rpm:>10.0f}  {J:>8.4f}  "
              f"{T:>8.3f}  {err:>+8.3f}")

        if np.isnan(T):
            print(f"\n  J = {J:.4f} is outside the table at RPM = {rpm:.0f}. "
                  f"Target may be unreachable at this cruise speed.")
            return None

        if abs(T - target_thrust) <= thrust_tol:
            print(f"\n  Converged in {i} iteration(s).")
            print(f"\n{chr(8212)*60}")
            print(f"  Operating point:")
            print(f"    RPM        = {rpm:.0f} RPM")
            print(f"    J          = {J:.4f}")
            print(f"    Thrust     = {T:.3f} N   (target: {target_thrust:.2f} N)")
            print(f"    Efficiency = {eff:.4f}")
            print(f"    Cp         = {Cp:.4f}")
            print(f"{chr(8212)*60}\n")
            return {"RPM": rpm, "J": J, "Thrust": T, "Efficiency": eff, "Cp": Cp}

        rpm += direction * rpm_step

    print(f"\n  Did not converge within {max_iter} iterations.")
    return None


if __name__ == "__main__":
    result = solve(
        csv_file      = CSV_FILE,
        target_thrust = TARGET_THRUST,
        cruise_speed  = CRUISE_SPEED,
        diameter      = DIAMETER,
        rpm_init      = RPM_INIT,
        rpm_tol       = RPM_TOL,
        thrust_tol    = THRUST_TOL,
        max_iter      = MAX_ITER,
        rpm_step      = RPM_STEP,
    )
