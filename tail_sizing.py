import numpy as np
import matplotlib.pyplot as plt
import initial_sizing as ins

Sw = ins.Sw
bw = ins.b
cw = ins.c

Vv = 0.04 # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5
Vh = 0.50 # https://icas.org/icas_archive/ICAS2022/data/papers/ICAS2022_0383_paper.pdf p.5

SLv = Vv * Sw * bw
SLh = Vh * Sw * cw

