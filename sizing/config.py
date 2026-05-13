# sizing/config.py

from dataclasses import dataclass
from sizing.materials import CF_PLA, CFRP, EPP

@dataclass
class MaterialsConfig:
    fuselage: object = CF_PLA()
    wing: object = EPP()
    rod: object = CFRP()
    tail: object = EPP()