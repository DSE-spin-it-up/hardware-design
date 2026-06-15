"""Estimate cruise propeller tonal noise with the Gutin loading-noise formula.

The implementation uses the Gutin formula for the RMS pressure of the q-th
blade-passing harmonic:

    p_q = (q n omega) / (2 sqrt(2) pi c L)
          * (-T cos(beta) + c Q / (omega R^2))
          * J_qn(q n sin(beta) V / c)

where n is blade count, omega is angular speed, c is speed of sound, L is
observer distance, T is thrust, Q is torque, R is prop radius, V is tip speed,
and beta is observer angle from the propeller axis. SPL is reported against
20 uPa.

Run:
    python gutin_noise.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from scipy.special import jv

from propulsion import propeller_solver


P_REF = 20e-6
GAMMA_AIR = 1.4
R_AIR = 287.05

# ---------------------------------------------------------------------------
# Edit these values when they cannot be pulled from config/design files.
# ---------------------------------------------------------------------------
CONFIG_FILE = "config.yaml"
DESIGN_FILE = "final_design_values.json"
PROPELLER_BLADES = 2
HARMONICS_TO_PRINT = 5
OBSERVER_DISTANCE_M = 100.0
OBSERVER_ANGLE_DEG = 90.0
AIR_TEMPERATURE_C = 15.0
AIR_DENSITY_KG_M3 = 1.225
SUM_ALL_PROPS_AND_DRONES = True


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    thrust_per_prop_n: float
    torque_per_prop_nm: float
    rpm: float
    tip_mach: float
    blade_passing_frequency_hz: float


def sound_speed_mps(temperature_c: float) -> float:
    """Speed of sound in dry air at the given temperature."""
    return math.sqrt(GAMMA_AIR * R_AIR * (temperature_c + 273.15))


def diameter_from_csv_name(csv_path: str) -> float:
    """Parse prop diameter from names like ``12x8E_performance.csv``."""
    stem = Path(csv_path.replace("\\", "/")).name
    match = re.match(r"(?P<diameter>\d+(?:\.\d+)?)x", stem)
    if not match:
        raise ValueError(f"Could not parse propeller diameter from {csv_path!r}")
    return float(match.group("diameter")) * 0.0254


def spl_from_pressure(pressure_rms_pa: float) -> float:
    """Convert RMS acoustic pressure to dB SPL."""
    if pressure_rms_pa <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(pressure_rms_pa / P_REF)


def shaft_power_from_cp(cp: float, air_density_kg_m3: float, n_rps: float, diameter_m: float) -> float:
    """Propeller shaft power from Cp."""
    return cp * air_density_kg_m3 * n_rps**3 * diameter_m**5


def gutin_harmonic_pressure(
    *,
    harmonic: int,
    blades: int,
    thrust_n: float,
    torque_nm: float,
    diameter_m: float,
    rpm: float,
    observer_distance_m: float,
    observer_angle_deg: float,
    speed_of_sound_mps: float,
) -> float:
    """Return RMS pressure [Pa] for one harmonic using the pictured formula."""
    beta = math.radians(observer_angle_deg)
    omega_rad_s = 2.0 * math.pi * rpm / 60.0
    radius_m = diameter_m / 2.0
    tip_speed_mps = omega_rad_s * radius_m
    order = harmonic * blades
    argument = order * math.sin(beta) * tip_speed_mps / speed_of_sound_mps
    loading = (
        -thrust_n * math.cos(beta)
        + speed_of_sound_mps * torque_nm / (omega_rad_s * radius_m**2)
    )
    pressure = (
        order
        * omega_rad_s
        / (2.0 * math.sqrt(2.0) * math.pi * speed_of_sound_mps * observer_distance_m)
        * loading
        * jv(order, argument)
    )
    return abs(float(pressure))


def combined_spl(levels_db: list[float]) -> float:
    """Energy-sum SPL values."""
    energy = sum(10.0 ** (level / 10.0) for level in levels_db if math.isfinite(level))
    if energy <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(energy)


def load_inputs(config_path: Path, design_path: Path) -> tuple[dict, dict]:
    with config_path.open() as f:
        config = yaml.safe_load(f)
    with design_path.open() as f:
        design = json.load(f)
    return config, design


def resolve_repo_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def solve_cruise_op(
    *,
    csv_prop: Path,
    thrust_per_prop_n: float,
    cruise_speed_mps: float,
    diameter_m: float,
    rpm_init: float,
    rpm_tol: float,
    thrust_tol: float,
    max_iter: int,
    rpm_step: float,
) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        result = propeller_solver.solve(
            str(csv_prop),
            thrust_per_prop_n,
            cruise_speed_mps,
            diameter_m,
            rpm_init,
            rpm_tol,
            thrust_tol,
            max_iter,
            rpm_step,
        )
    if result is None:
        raise RuntimeError("Cruise propeller operating point did not converge.")
    return result


def cruise_operating_point(
    *,
    config: dict,
    design: dict,
    repo_root: Path,
    temperature_c: float,
    air_density_kg_m3: float,
) -> tuple[OperatingPoint, float]:
    prop = config["propulsion"]
    csv_prop = resolve_repo_path(prop["csv_prop"], repo_root)
    diameter_m = prop.get("D_prop") or diameter_from_csv_name(str(csv_prop))
    n_props = int(design.get("n_props_per_drone", prop["n_props"]))
    cruise_speed_mps = float(design.get("cruise_speed_mps", config["sizing"]["V_cruise"]))
    a_mps = sound_speed_mps(temperature_c)

    cruise_thrust_per_prop = float(design["cruise_drag_per_drone_N"]) / n_props
    cruise = solve_cruise_op(
        csv_prop=csv_prop,
        thrust_per_prop_n=cruise_thrust_per_prop,
        cruise_speed_mps=cruise_speed_mps,
        diameter_m=diameter_m,
        rpm_init=float(prop["cruise_rpm_init"]),
        rpm_tol=float(prop["rpm_tol"]),
        thrust_tol=float(prop["thrust_tol"]),
        max_iter=int(prop["max_iter"]),
        rpm_step=float(prop["rpm_step"]),
    )

    rpm = float(cruise["RPM"])
    n_rps = rpm / 60.0
    omega_rad_s = 2.0 * math.pi * n_rps
    shaft_power_w = shaft_power_from_cp(
        float(cruise["Cp"]),
        air_density_kg_m3,
        n_rps,
        diameter_m,
    )
    torque_nm = shaft_power_w / omega_rad_s
    tip_mach = (math.pi * diameter_m * rpm / 60.0) / a_mps
    return (
        OperatingPoint(
            name="cruise",
            thrust_per_prop_n=cruise_thrust_per_prop,
            torque_per_prop_nm=torque_nm,
            rpm=rpm,
            tip_mach=tip_mach,
            blade_passing_frequency_hz=0.0,
        ),
        diameter_m,
    )


def source_count(config: dict, design: dict, *, single_prop: bool) -> int:
    if single_prop:
        return 1
    n_props = int(design.get("n_props_per_drone", config["propulsion"]["n_props"]))
    n_drones = int(design.get("n_drones", config["sizing"]["n_drones"]))
    return n_props * n_drones


def print_noise_table(
    *,
    point: OperatingPoint,
    diameter_m: float,
    blades: int,
    harmonics: int,
    observer_distance_m: float,
    observer_angle_deg: float,
    temperature_c: float,
    air_density_kg_m3: float,
    n_sources: int,
) -> None:
    a_mps = sound_speed_mps(temperature_c)
    print("\n--- Gutin Cruise Propeller Tonal Noise ---")
    print(f"Prop diameter       : {diameter_m:.4f} m ({diameter_m / 0.0254:.2f} in)")
    print(f"Blade count         : {blades}")
    print(f"Observer distance   : {observer_distance_m:.2f} m")
    print(f"Observer angle      : {observer_angle_deg:.1f} deg from prop axis")
    print(f"Temperature         : {temperature_c:.1f} C (a = {a_mps:.1f} m/s)")
    print(f"Air density         : {air_density_kg_m3:.3f} kg/m^3")
    print(f"Equal sources summed: {n_sources}")
    print(f"Thrust / prop       : {point.thrust_per_prop_n:.2f} N")
    print(f"Torque / prop       : {point.torque_per_prop_nm:.3f} N m")
    print(f"RPM                 : {point.rpm:.0f}")
    print(f"Tip Mach            : {point.tip_mach:.3f}")

    bpf = blades * point.rpm / 60.0
    print(f"BPF                 : {bpf:.1f} Hz")
    print("\nHarmonic   Freq [Hz]   SPL 1 prop [dB]   SPL summed [dB]")

    one_prop_levels = []
    summed_levels = []
    for harmonic in range(1, harmonics + 1):
        pressure = gutin_harmonic_pressure(
            harmonic=harmonic,
            blades=blades,
            thrust_n=point.thrust_per_prop_n,
            torque_nm=point.torque_per_prop_nm,
            diameter_m=diameter_m,
            rpm=point.rpm,
            observer_distance_m=observer_distance_m,
            observer_angle_deg=observer_angle_deg,
            speed_of_sound_mps=a_mps,
        )
        spl_one = spl_from_pressure(pressure)
        spl_sum = spl_one + 10.0 * math.log10(n_sources)
        one_prop_levels.append(spl_one)
        summed_levels.append(spl_sum)
        print(
            f"{harmonic:>8d}"
            f"   {harmonic * bpf:>9.1f}"
            f"   {spl_one:>15.1f}"
            f"   {spl_sum:>15.1f}"
        )

    print(f"\nTotal tones, 1 prop : {combined_spl(one_prop_levels):.1f} dB SPL")
    print(f"Total tones, summed : {combined_spl(summed_levels):.1f} dB SPL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(CONFIG_FILE))
    parser.add_argument("--design", type=Path, default=Path(DESIGN_FILE))
    parser.add_argument("--blades", type=int, default=PROPELLER_BLADES)
    parser.add_argument("--harmonics", type=int, default=HARMONICS_TO_PRINT)
    parser.add_argument(
        "--distance",
        type=float,
        default=OBSERVER_DISTANCE_M,
        help="Observer distance [m]",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=OBSERVER_ANGLE_DEG,
        help="Observer angle from propeller axis [deg]",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=AIR_TEMPERATURE_C,
        help="Air temperature [C]",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=AIR_DENSITY_KG_M3,
        help="Air density [kg/m^3]",
    )
    parser.add_argument(
        "--single-prop",
        action="store_true",
        default=not SUM_ALL_PROPS_AND_DRONES,
        help="Do not add the +10log10(N) correction for all props and drones.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = resolve_repo_path(str(args.config), repo_root)
    design_path = resolve_repo_path(str(args.design), repo_root)
    config, design = load_inputs(config_path, design_path)
    point, diameter_m = cruise_operating_point(
        config=config,
        design=design,
        repo_root=repo_root,
        temperature_c=args.temperature,
        air_density_kg_m3=args.density,
    )
    print_noise_table(
        point=point,
        diameter_m=diameter_m,
        blades=args.blades,
        harmonics=args.harmonics,
        observer_distance_m=args.distance,
        observer_angle_deg=args.angle,
        temperature_c=args.temperature,
        air_density_kg_m3=args.density,
        n_sources=source_count(config, design, single_prop=args.single_prop),
    )


if __name__ == "__main__":
    main()
