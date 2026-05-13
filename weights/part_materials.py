from dataclasses import dataclass
from structures.materials import CF_PLA, CFRP, EPP

@dataclass
class PartMaterials:
    fuselage: object = CF_PLA()
    wing: object = EPP()
    rod: object = CFRP()
    tail: object = EPP()