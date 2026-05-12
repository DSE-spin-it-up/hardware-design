from dataclasses import dataclass, fields

import numpy as np

from sizing.initial_sizing import SizingResult


@dataclass
class FuselageInputs:
    # Battery cell envelope [m]
    battery_length: float = 0.212
    battery_width: float = 0.090
    battery_height: float = 0.060
    # Geometric margins
    housing_factor: float = 1.5  # axial battery housing factor
    casing_factor: float = 1.1   # all-around casing margin


@dataclass
class FuselageResult:
    inputs: FuselageInputs
    length: float       # [m]
    width: float        # [m]
    height: float       # [m]
    d_eq: float         # equivalent diameter [m]
    fineness: float     # length / d_eq [-]
    Swet: float         # wetted area [m^2]


def run(
    sizing: SizingResult,
    inputs: FuselageInputs | None = None,
) -> FuselageResult:
    if inputs is None:
        inputs = FuselageInputs()
    i = inputs

    length = max(
        i.casing_factor * sizing.c_root,
        i.battery_length * i.housing_factor * i.casing_factor,
    )
    width = i.battery_width * i.casing_factor
    height = i.battery_height * i.casing_factor

    d_eq = np.sqrt(width * height)
    fineness = length / d_eq
    Swet = 2 * (length * width + length * height + width * height)

    return FuselageResult(
        inputs=inputs,
        length=length,
        width=width,
        height=height,
        d_eq=d_eq,
        fineness=fineness,
        Swet=Swet,
    )


def summary(r: FuselageResult) -> None:
    print("\n--- Fuselage ---")
    print(f"  Length               : {r.length:.4f}  m")
    print(f"  Width                : {r.width:.4f}  m")
    print(f"  Height               : {r.height:.4f}  m")
    print(f"  Fineness ratio       : {r.fineness:.3f}")
    print(f"  Wetted area          : {r.Swet:.4f}  m²")


# Default run at module load — exposes inputs+result fields as module attributes.
from sizing import initial_sizing as _is  # noqa: E402
_default = run(_is._default)
for _f in fields(FuselageInputs):
    globals()[_f.name] = getattr(_default.inputs, _f.name)
for _f in fields(FuselageResult):
    if _f.name != "inputs":
        globals()[_f.name] = getattr(_default, _f.name)


if __name__ == "__main__":
    summary(_default)
