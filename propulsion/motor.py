import numpy as np

KV = 477
n = 13375
V_emf = n / KV
Q = 1.04 
R_w = 84/1000
I0 = 1.8
Kt = 60 / (2 * np.pi * KV)
I = (Q / Kt) + I0
V = I * R_w + V_emf

print(I)
print(V)
print(V * I)