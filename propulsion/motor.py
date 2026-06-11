import numpy as np

KV = 700
n = 8600
V_emf = n / KV
Q = 0.502
R_w = 0.0327
I0 = 1.8
Kt = 60 / (2 * np.pi * KV)
I = (Q / Kt) + I0
V = I * R_w + V_emf

print(I)
print(V)
print(V * I)