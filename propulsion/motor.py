import numpy as np

KV = 477
n = 8550 #8550 11300 13375
V_emf = n / KV
Q = 0.502
R_w = 40/1000
I0 = 1.1
Kt = 60 / (2 * np.pi * KV)
I = (Q / Kt) + I0
V = I * R_w + V_emf

print(I)
print(V)
print(V * I)