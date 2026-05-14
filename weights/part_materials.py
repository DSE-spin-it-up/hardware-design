from dataclasses import dataclass, field

from structures.materials import CFRP, CF_PLA, EPP, PLA, Material, Wood

MATERIAL_REGISTRY: dict[str, type[Material]] = {
    "CFRP": CFRP,
    "EPP": EPP,
    "CF_PLA": CF_PLA,
    "PLA": PLA,
    "Wood": Wood,
}


def resolve_material(name: str) -> Material:
    try:
        cls = MATERIAL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown material {name!r}. Options: {sorted(MATERIAL_REGISTRY)}"
        ) from exc
    return cls()


@dataclass
class PartMaterials:
    fuselage: Material = field(default_factory=CF_PLA)
    wing: Material = field(default_factory=EPP)
    rod: Material = field(default_factory=CFRP)
    tail: Material = field(default_factory=EPP)

    @classmethod
    def from_names(cls, names: dict[str, str]) -> "PartMaterials":
        return cls(**{part: resolve_material(name) for part, name in names.items()})
