import numpy as np
import initial_sizing as ins

Sw = ins.Sw # [m^2]
bw = ins.b # [m]
cw = ins.c # [m]
tc = ins.t_over_c_root # [-]
rho = ins.foam_density # [kg/m^3]

Vv = 0.04 # [-] https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
Vh = 0.50 # [-] https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5

bh = bw * 0.365445026178 # [m] Desmos
ARt = 4.5 # [-] https://www.fmsg-alling.de/wp-content/uploads/2013/09/V-Leitwerke.pdf

Sh = bh**2 / ARt # [m^2]
L = Vh * Sw * cw / Sh # [m]
Sv = Vv * Sw * bw / L # [m^2]
bv = np.sqrt(2 * ARt * Sw) / 2 # [m]

St = Sh + Sv # [m^2]
bt = np.sqrt(bh**2 + bv**2) # [m]

lam = 1 # [-]

def ct(yb):
    ct = (2 * St) / ((1 + lam) * bt) * (1 - (1 - lam) * (2 * yb))
    return ct

ct = ct(0.5) # [m]

tt = ct * tc # [m]
mt = St * tt * rho # [kg]

print("Horizontal Tail Area:", Sh)
print("Vertical Tail Area:", Sv)
print("Tail Length:", L)
print("Tail Chord:", ct)
print("Tail Mass:", mt)