import numpy as np
import initial_sizing as ins

Sw = ins.Sw
bw = ins.b
cw = ins.c

Vv = 0.04 # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
Vh = 0.50 # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5

bh = bw * 0.365445026178 # Desmos
ARt = 4.5 # https://www.fmsg-alling.de/wp-content/uploads/2013/09/V-Leitwerke.pdf

Sh = bh**2 * ARt
L = Vh * Sw * cw / Sh
Sv = Vv * Sw * bw / L

print("Horizontal Tail Area:", Sh)
print("Vertical Tail Area:", Sv)
print("Tail Length:", L)