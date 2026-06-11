import numpy as np

KV = 600
n = 13375
V_emf = n / KV
Q = 1.04 
R_w = 0.033
I0 = 2.4
Kt = 60 / (2 * np.pi * KV)
I = (Q / Kt) + I0
V = I * R_w + V_emf

print(I)
print(V)
print(V * I)