import numpy as np

KV = 350 #350 477
n = 13375 #8550 11300 13375
V_emf = n / KV
Q = 1.02
R_w = 40/1000 # 40 84
I0 = 1.1 # 1.1 0.96
Kt = 60 / (2 * np.pi * KV)
I = (Q / Kt) + I0
V = I * R_w + V_emf

print(I)
print(V)
print(V * I)