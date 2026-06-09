"""Mass moment of inertia estimate for the converged aircraft.

The project already computes component masses plus longitudinal and vertical
CG locations. This module adds lateral placement assumptions and applies the
parallel-axis theorem to estimate the moments of inertia about axes through the
overall aircraft CG.

Coordinate convention:
    x: longitudinal, positive aft from LEMAC [m]
    y: lateral/spanwise, positive starboard [m]
    z: vertical, positive upward from the side-view datum [m]

Returned inertias are about CG-centred axes parallel to that coordinate frame:
    Ixx: roll inertia about the longitudinal axis [kg*m^2]
    Iyy: pitch inertia about the lateral axis [kg*m^2]
    Izz: yaw inertia about the vertical axis [kg*m^2]
"""
from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import dataclass

from sizing.fuselage import FuselageResult
from sizing.wing import SizingResult
from structures.rods import RodResult
from weights.mass import compute_y_cg


@dataclass(frozen=True)
class PointMass:
    """A component represented by mass, CG, and optional centroidal inertia."""

    name: str
    mass: float
    x: float
    y: float
    z: float
    ixx_centroid: float = 0.0
    iyy_centroid: float = 0.0
    izz_centroid: float = 0.0


@dataclass(frozen=True)
class InertiaResult:
    """Mass moment of inertia result about the overall aircraft CG."""

    mass: float
    x_cg: float
    y_cg: float
    z_cg: float
    ixx: float
    iyy: float
    izz: float
    components: tuple[PointMass, ...]


def _box_inertia(mass: float, length_x: float, width_y: float, height_z: float) -> tuple[float, float, float]:
    """Centroidal inertia of a rectangular box with dimensions along x, y, z."""
    return (
        mass * (width_y**2 + height_z**2) / 12.0,
        mass * (length_x**2 + height_z**2) / 12.0,
        mass * (length_x**2 + width_y**2) / 12.0,
    )


def _rod_along_x_inertia(mass: float, length: float) -> tuple[float, float, float]:
    """Thin rod inertia for a rod aligned with the x-axis."""
    return (0.0, mass * length**2 / 12.0, mass * length**2 / 12.0)


def _rod_along_y_inertia(mass: float, length: float) -> tuple[float, float, float]:
    """Thin rod inertia for a rod aligned with the y-axis."""
    return (mass * length**2 / 12.0, 0.0, mass * length**2 / 12.0)


def _rod_along_z_inertia(mass: float, length: float) -> tuple[float, float, float]:
    """Thin rod inertia for a rod aligned with the z-axis."""
    return (mass * length**2 / 12.0, mass * length**2 / 12.0, 0.0)


def _thin_plate_xy_inertia(mass: float, length_x: float, width_y: float) -> tuple[float, float, float]:
    """Centroidal inertia of a thin plate in the x-y plane."""
    return (
        mass * width_y**2 / 12.0,
        mass * length_x**2 / 12.0,
        mass * (length_x**2 + width_y**2) / 12.0,
    )


def _thin_plate_xz_inertia(mass: float, length_x: float, height_z: float) -> tuple[float, float, float]:
    """Centroidal inertia of a thin plate in the x-z plane."""
    return (
        mass * height_z**2 / 12.0,
        mass * (length_x**2 + height_z**2) / 12.0,
        mass * length_x**2 / 12.0,
    )


def _append_if_positive(components: list[PointMass], component: PointMass) -> None:
    if component.mass > 0.0:
        components.append(component)


def _add_front_and_rear_propulsion(
    components: list[PointMass],
    *,
    name: str,
    total_mass: float,
    n_props: int,
    front_x: float,
    rear_x: float,
    front_z: float,
    rear_z: float,
    wing_span: float,
) -> None:
    """Place propulsion mass at two front wing stations and remaining rear stations.

    The existing CG code assumes the three-prop layout has two front propulsion
    units on the wing and one rear unit on the tail. For other prop counts this
    keeps the first two as symmetric front units and puts the rest on the
    centreline rear station.
    """
    if total_mass <= 0.0 or n_props <= 0:
        return

    mass_each = total_mass / n_props
    front_count = min(2, n_props)
    rear_count = max(n_props - front_count, 0)
    front_y = 0.25 * wing_span

    if front_count == 1:
        _append_if_positive(components, PointMass(f"{name}_front", mass_each, front_x, 0.0, front_z))
    elif front_count == 2:
        _append_if_positive(components, PointMass(f"{name}_front_left", mass_each, front_x, -front_y, front_z))
        _append_if_positive(components, PointMass(f"{name}_front_right", mass_each, front_x, front_y, front_z))

    for idx in range(rear_count):
        suffix = "rear" if rear_count == 1 else f"rear_{idx + 1}"
        _append_if_positive(components, PointMass(f"{name}_{suffix}", mass_each, rear_x, 0.0, rear_z))


def build_components(
    *,
    sizing: SizingResult,
    fus: FuselageResult,
    structure: RodResult,
    masses: dict[str, float],
    cg: dict[str, float],
    z_cg: dict[str, float],
    n_props: int,
) -> tuple[PointMass, ...]:
    """Build component inertial model from the pipeline outputs."""
    components: list[PointMass] = []

    fus_i = _box_inertia(masses.get("fuselage", 0.0), fus.length, fus.width, fus.height)
    _append_if_positive(
        components,
        PointMass("fuselage", masses.get("fuselage", 0.0), cg["fuselage"], 0.0, z_cg["fuselage"], *fus_i),
    )

    batt_i = _box_inertia(
        masses.get("battery", 0.0),
        fus.battery_length,
        fus.battery_width,
        fus.battery_height,
    )
    _append_if_positive(
        components,
        PointMass("battery", masses.get("battery", 0.0), cg["battery"], 0.0, z_cg["battery"], *batt_i),
    )

    _add_front_and_rear_propulsion(
        components,
        name="motor",
        total_mass=masses.get("motors", 0.0),
        n_props=n_props,
        front_x=cg["motors"],
        rear_x=cg.get("motor_back", cg["motors"]),
        front_z=z_cg["motors"],
        rear_z=z_cg.get("motor_back", z_cg["motors"]),
        wing_span=sizing.inputs.b,
    )
    _add_front_and_rear_propulsion(
        components,
        name="prop",
        total_mass=masses.get("props", 0.0),
        n_props=n_props,
        front_x=cg["motors"],
        rear_x=cg.get("motor_back", cg["motors"]),
        front_z=z_cg["motors"],
        rear_z=z_cg.get("motor_back", z_cg["motors"]),
        wing_span=sizing.inputs.b,
    )

    wing_i = _thin_plate_xy_inertia(masses.get("wing", 0.0), sizing.c_root, sizing.inputs.b)
    _append_if_positive(components, PointMass("wing", masses.get("wing", 0.0), cg["wing"], 0.0, z_cg["wing"], *wing_i))

    rib_i = _thin_plate_xy_inertia(masses.get("ribs", 0.0), sizing.c_root, 0.0)
    _append_if_positive(components, PointMass("ribs", masses.get("ribs", 0.0), cg.get("ribs", cg["wing"]), 0.0, z_cg.get("ribs", z_cg["wing"]), *rib_i))

    spar_i = _rod_along_y_inertia(masses.get("rod_spar", 0.0), sizing.inputs.b)
    _append_if_positive(components, PointMass("rod_spar", masses.get("rod_spar", 0.0), cg["rod_spar"], 0.0, z_cg["rod_spar"], *spar_i))

    aileron_i = _rod_along_y_inertia(masses.get("rod_aileron", 0.0), sizing.inputs.b)
    _append_if_positive(components, PointMass("rod_aileron", masses.get("rod_aileron", 0.0), cg["rod_aileron"], 0.0, z_cg["rod_aileron"], *aileron_i))

    boom_i = _rod_along_x_inertia(masses.get("tail_rod", 0.0), sizing.L_boom)
    _append_if_positive(components, PointMass("tail_rod", masses.get("tail_rod", 0.0), cg["tail_rod"], 0.0, z_cg["tail_rod"], *boom_i))

    ht_i = _thin_plate_xy_inertia(masses.get("hor_tail", 0.0), sizing.ch, sizing.bh)
    _append_if_positive(components, PointMass("hor_tail", masses.get("hor_tail", 0.0), cg["tail"], 0.0, z_cg["hor_tail"], *ht_i))

    vt_i = _thin_plate_xz_inertia(masses.get("ver_tail", 0.0), sizing.cv, sizing.bv)
    _append_if_positive(components, PointMass("ver_tail", masses.get("ver_tail", 0.0), cg["tail"], 0.0, z_cg["ver_tail"], *vt_i))

    ht_spar_i = _rod_along_y_inertia(masses.get("ht_spar", 0.0), sizing.bh)
    _append_if_positive(components, PointMass("ht_spar", masses.get("ht_spar", 0.0), cg["ht_spar"], 0.0, z_cg.get("hor_tail", z_cg["tail_rod"]), *ht_spar_i))

    ht_rud_i = _rod_along_y_inertia(masses.get("ht_rud", 0.0), sizing.bh)
    _append_if_positive(components, PointMass("ht_rud", masses.get("ht_rud", 0.0), cg["ht_rud"], 0.0, z_cg.get("hor_tail", z_cg["tail_rod"]), *ht_rud_i))

    vt_spar_i = _rod_along_z_inertia(masses.get("vt_spar", 0.0), sizing.bv)
    _append_if_positive(components, PointMass("vt_spar", masses.get("vt_spar", 0.0), cg["vt_spar"], 0.0, z_cg["vt_spar"], *vt_spar_i))

    vt_rud_i = _rod_along_z_inertia(masses.get("vt_rud", 0.0), sizing.bv)
    _append_if_positive(components, PointMass("vt_rud", masses.get("vt_rud", 0.0), cg["vt_rud"], 0.0, z_cg["vt_rud"], *vt_rud_i))

    pvc_i = _box_inertia(
        masses.get("pvc_tubes", 0.0),
        fus.box_length,
        fus.box_width,
        fus.structural_tube_outer_diameter,
    )
    _append_if_positive(components, PointMass("pvc_tubes", masses.get("pvc_tubes", 0.0), cg["pvc_tubes"], 0.0, z_cg["pvc_tubes"], *pvc_i))

    _append_if_positive(components, PointMass("servos", masses.get("servos", 0.0), cg["servos"], 0.0, z_cg["servos"]))
    _append_if_positive(components, PointMass("sensors", masses.get("sensors", 0.0), cg["sensors"], 0.0, z_cg["sensors"]))
    _append_if_positive(components, PointMass("wiring", masses.get("wiring", 0.0), cg["wiring_signal"], 0.0, z_cg["wiring_signal"]))

    sheet_wing_i = _thin_plate_xy_inertia(masses.get("glass_sheet_wing", 0.0), sizing.c_root, sizing.inputs.b)
    _append_if_positive(components, PointMass("glass_sheet_wing", masses.get("glass_sheet_wing", 0.0), cg["wing"], 0.0, z_cg["glass_sheet_wing"], *sheet_wing_i))

    sheet_ht_i = _thin_plate_xy_inertia(masses.get("glass_sheet_tail_h", 0.0), sizing.ch, sizing.bh)
    _append_if_positive(components, PointMass("glass_sheet_tail_h", masses.get("glass_sheet_tail_h", 0.0), cg["tail"], 0.0, z_cg["glass_sheet_tail_h"], *sheet_ht_i))

    sheet_vt_i = _thin_plate_xz_inertia(masses.get("glass_sheet_tail_v", 0.0), sizing.cv, sizing.bv)
    _append_if_positive(components, PointMass("glass_sheet_tail_v", masses.get("glass_sheet_tail_v", 0.0), cg["tail"], 0.0, z_cg["glass_sheet_tail_v"], *sheet_vt_i))

    modeled_mass = sum(component.mass for component in components)
    mass_margin = masses.get("total", 0.0) - modeled_mass
    _append_if_positive(
        components,
        PointMass("mass_margin", mass_margin, cg["overall"], 0.0, z_cg["overall"]),
    )

    return tuple(components)


def calculate_mass_moment_of_inertia(
    *,
    sizing: SizingResult,
    fus: FuselageResult,
    structure: RodResult,
    masses: dict[str, float],
    cg: dict[str, float],
    z_cg: dict[str, float],
    n_props: int,
) -> InertiaResult:
    """Calculate Ixx, Iyy, and Izz about the aircraft CG."""
    components = build_components(
        sizing=sizing,
        fus=fus,
        structure=structure,
        masses=masses,
        cg=cg,
        z_cg=z_cg,
        n_props=n_props,
    )

    mass = sum(component.mass for component in components)
    if mass <= 0.0:
        raise ValueError("Cannot calculate inertia with zero total component mass.")

    x_cg = sum(component.mass * component.x for component in components) / mass
    y_cg = sum(component.mass * component.y for component in components) / mass
    z_cg_total = sum(component.mass * component.z for component in components) / mass

    ixx = 0.0
    iyy = 0.0
    izz = 0.0
    for component in components:
        dx = component.x - x_cg
        dy = component.y - y_cg
        dz = component.z - z_cg_total
        ixx += component.ixx_centroid + component.mass * (dy**2 + dz**2)
        iyy += component.iyy_centroid + component.mass * (dx**2 + dz**2)
        izz += component.izz_centroid + component.mass * (dx**2 + dy**2)

    return InertiaResult(
        mass=mass,
        x_cg=x_cg,
        y_cg=y_cg,
        z_cg=z_cg_total,
        ixx=ixx,
        iyy=iyy,
        izz=izz,
        components=components,
    )


def calculate_from_pipeline(*, verbose_pipeline: bool = False) -> InertiaResult:
    """Run the design pipeline and calculate the final aircraft inertia."""
    from pipeline import config, loop

    if verbose_pipeline:
        result = loop.run_pipeline(config)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = loop.run_pipeline(config)
    z_cg = compute_y_cg(
        result.sizing,
        result.fus,
        result.struct,
        result.masses,
        result.airfoil,
        cg=result.cg,
    )
    return calculate_mass_moment_of_inertia(
        sizing=result.sizing,
        fus=result.fus,
        structure=result.struct,
        masses=result.masses,
        cg=result.cg,
        z_cg=z_cg,
        n_props=result.propulsion.inputs.n_props,
    )


def print_inertia(result: InertiaResult, *, show_components: bool = False) -> None:
    """Print a compact inertia report."""
    print("\nMass moment of inertia about aircraft CG")
    print(f"  mass used : {result.mass:.3f} kg")
    print(f"  CG        : x={result.x_cg:.4f} m, y={result.y_cg:.4f} m, z={result.z_cg:.4f} m")
    print(f"  Ixx roll  : {result.ixx:.4f} kg*m^2")
    print(f"  Iyy pitch : {result.iyy:.4f} kg*m^2")
    print(f"  Izz yaw   : {result.izz:.4f} kg*m^2")

    if show_components:
        print("\nComponent model")
        for component in result.components:
            print(
                f"  {component.name:20s}"
                f" m={component.mass:7.3f} kg"
                f" x={component.x:8.4f}"
                f" y={component.y:8.4f}"
                f" z={component.z:8.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate aircraft mass moment of inertia.")
    parser.add_argument("--components", action="store_true", help="Print the component inertial model")
    parser.add_argument("--verbose-pipeline", action="store_true", help="Show the full design pipeline output")
    args = parser.parse_args()

    inertia = calculate_from_pipeline(verbose_pipeline=args.verbose_pipeline)
    print_inertia(inertia, show_components=args.components)


if __name__ == "__main__":
    main()
