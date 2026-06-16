import numpy as np

KV = 700 #350 477
n = 10000 #8550 11300 13375
V_emf = n / KV
Q = 0.6
R_w = 33/1000 # 40 84
I0 = 3.4 # 1.1 0.96
Kt = 60 / (2 * np.pi * KV)
I = (Q / Kt) + I0
V = I * R_w + V_emf
print(V)
print(I)

print(V/24)