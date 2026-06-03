from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sizing.wing import SizingResult
from structures.materials import EPP


@dataclass
class FuselageInputs:
    # Off-the-shelf battery cell envelope [m] — used when AR_lw == 0 or AR_lh == 0.
    battery_length: float = 0.212
    battery_width: float = 0.090
    battery_height: float = 0.060
    # Battery aspect ratios. If both > 0, dimensions are recomputed from the
    # required battery volume (from propulsion sizing) and these ratios,
    # overriding battery_length/width/height above.
    AR_lw: float = 0.0  # length / width
    AR_lh: float = 0.0  # length / height
    # Geometric margin applied uniformly to all internal components (battery
    # and tubes).  A single factor keeps the EPP foam walls consistent on
    # every face: each internal dimension is multiplied by casing_factor so
    # the wall thickness is roughly (casing_factor - 1) / 2 × dimension.
    # 1.15 → ~7.5 % wall on each side; 1.25 → ~12.5 %.
    casing_factor: float = 1.15  # [-] all-around margin for EPP foam walls
    casing_thickness: float = 0.001  # [m] minimum hard wall thickness
    # How far the tube extends aft of the aileron hinge to grip the tail boom.
    tube_tail_overlap: float = 0.03  # [m]
    # EPP tearout prevention
    load_factor: float = 3.0        # [-] landing impact load factor
    g: float = 9.81                 # [m/s²]
    min_foam_floor: float = 0.005   # [m] absolute minimum foam floor thickness


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    length: float       # [m]
    width: float        # [m]
    height: float       # [m]
    d_eq: float         # equivalent diameter [m]
    fineness: float     # length / d_eq [-]
    Swet: float         # wetted area [m²]
    battery_length: float  # [m] battery dimension actually used
    battery_width: float   # [m]
    battery_height: float  # [m]
    volume_shell: float = 0.0           # [m³] structural material volume
    x_nose: float = 0.0                 # [m] fuselage nose x from LEMAC (negative = ahead of LE)
    battery_y_min: float = 0.0          # [m] lowest allowable battery bottom from fuselage floor
    foam_floor_thickness: float = 0.0   # [m] required foam thickness below battery


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
    battery_volume: float = 0.0,
    battery_x: float | None = None,
    battery_mass: float = 0.0,
    tube_back_x: float | None = None,
    tube_outer_diameter: float = 0.0,
) -> FuselageResult:
    """
    Fuselage geometry is driven entirely by the internal components:

      • Nose  : battery front face (battery_x − b_length / 2) minus one wall
      • Aft   : aileron hinge + tube_tail_overlap + one wall thickness.
                tube_back_x should be passed as x_rod_aileron + tube_tail_overlap.
                Falls back to sizing.c_root if not provided.
      • Height: max(battery height, tube diameter) × casing_factor + foam floor
      • Width : battery width × casing_factor
      • Floor : EPP shear / tearout check sets the minimum foam thickness
                below the battery; the fuselage floor sits that far below
                the battery bottom.

    Parameters
    ----------
    tube_back_x : x-position (from LEMAC) of the aft face of the tube.
                  Pass as x_rod_aileron + inputs.tube_tail_overlap from the pipeline.
                  Falls back to sizing.c_root if not provided.
    tube_outer_diameter : outer diameter of the largest tube (spar or aileron
                          rod), used to size fuselage height.
    """

    FILLED = True

    if inputs is None:
        inputs = FuselageInputs()
    i = inputs

    # ------------------------------------------------------------------ battery dims
    if i.AR_lw > 0 and i.AR_lh > 0:
        # L * (L/AR_lw) * (L/AR_lh) = V  =>  L = (V * AR_lw * AR_lh)^(1/3)
        b_length = (battery_volume * i.AR_lw * i.AR_lh) ** (1 / 3)
        b_width  = b_length / i.AR_lw
        b_height = b_length / i.AR_lh
    else:
        b_length = i.battery_length
        b_width  = i.battery_width
        b_height = i.battery_height

    # ------------------------------------------------------------------ EPP tearout check
    # Compute this first — foam_floor_thickness sets the fuselage floor position
    # which in turn feeds into the overall height.
    epp = EPP()
    bearing_area = b_length * b_width
    stress = (battery_mass * i.g * i.load_factor) / bearing_area if bearing_area > 0 else 0.0
    foam_floor_thickness = max(stress / epp.s_t, i.min_foam_floor)

    if stress / epp.s_t > i.min_foam_floor:
        print(f"  EPP floor: tearout governs        "
              f"(required {foam_floor_thickness*1e3:.1f} mm)")
    else:
        print(f"  EPP floor: min thickness governs  "
              f"(required {foam_floor_thickness*1e3:.1f} mm, "
              f"tearout would need {stress/epp.s_t*1e3:.1f} mm)")

    battery_y_min = foam_floor_thickness

    # ------------------------------------------------------------------ nose / aft x
    # Nose: one wall thickness ahead of the battery front face.
    if battery_x is not None:
        x_battery_front = battery_x - b_length / 2.0
    else:
        # Battery not yet placed — centre it in the root chord as a fallback.
        x_battery_front = sizing.c_root / 2.0 - b_length / 2.0

    x_nose = x_battery_front - i.casing_thickness

    # Aft: back of the tube (aileron hinge + overlap), plus one wall thickness.
    _tube_back = tube_back_x if tube_back_x is not None else sizing.c_root
    x_aft = _tube_back + i.casing_thickness

    length = x_aft - x_nose

    # ------------------------------------------------------------------ width
    # Battery width sets the cross-section; casing_factor gives uniform foam
    # walls on both sides.
    width = b_width * i.casing_factor

    # ------------------------------------------------------------------ height
    # The fuselage must clear whichever is taller: the battery or the tube.
    # casing_factor is applied to that governing dimension for a uniform wall.
    # The floor is raised by foam_floor_thickness (EPP shear), so the
    # external height includes that extra material at the bottom.
    inner_height = max(b_height, tube_outer_diameter) * i.casing_factor
    height = inner_height + foam_floor_thickness

    # ------------------------------------------------------------------ derived
    d_eq     = np.sqrt(width * height)
    fineness = length / d_eq
    Swet     = 2.0 * (length * width + length * height + width * height)

    if FILLED:
        volume_shell = length * width * height

    return FuselageResult(
        inputs=inputs,
        length=length,
        width=width,
        height=height,
        d_eq=d_eq,
        fineness=fineness,
        Swet=Swet,
        battery_length=b_length,
        battery_width=b_width,
        battery_height=b_height,
        volume_shell=volume_shell,
        x_nose=x_nose,
        battery_y_min=battery_y_min,
        foam_floor_thickness=foam_floor_thickness,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage ---")
    print(f"  Battery L × W × H    : "
          f"{r.battery_length:.4f} × {r.battery_width:.4f} × {r.battery_height:.4f}  m")
    print(f"  Length               : {r.length:.4f}  m")
    print(f"  Nose x (from LEMAC)  : {r.x_nose:.4f}  m")
    print(f"  Width                : {r.width:.4f}  m")
    print(f"  Height               : {r.height:.4f}  m")
    print(f"  Fineness ratio       : {r.fineness:.3f}")
    print(f"  Wetted area          : {r.Swet:.4f}  m²")
    print(f"  Shell volume         : {r.volume_shell:.6f}  m³")
    print(f"  Foam floor thickness : {r.foam_floor_thickness*1e3:.1f}  mm")
    print(f"  Battery y_min        : {r.battery_y_min*1e3:.1f}  mm")


if __name__ == "__main__":
    from sizing import wing
    from sizing.wing import SizingInputs
    summary(run(wing.run(SizingInputs())))