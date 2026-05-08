import numpy as np
import initial_sizing as ins

Sw = ins.Sw
bw = ins.b
cw = ins.c

Vv = 0.04 # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
Vh = 0.50 # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
Lh = 1.0 # [m]
Lv = 1.0 # [m]

Sv = Vv * Sw * bw / Lv
Sh = Vh * Sw * cw / Lh

vertical_tail_volume = Sv * ins.tc_root
horizontal_tail_volume = Sh * ins.tc_root

mass_vertical_tail = ins.foam_density * vertical_tail_volume
mass_horizontal_tail = ins.foam_density * horizontal_tail_volume

print(f"Vertical tail area: {Sv:.4f} m^2")
print(f"Horizontal tail area: {Sh:.4f} m^2")
print(f"Vertical tail volume: {vertical_tail_volume:.4f} m^3")
print(f"Horizontal tail volume: {horizontal_tail_volume:.4f} m^3")
print(f"Mass of vertical tail: {mass_vertical_tail:.4f} kg")
print(f"Mass of horizontal tail: {mass_horizontal_tail:.4f} kg")