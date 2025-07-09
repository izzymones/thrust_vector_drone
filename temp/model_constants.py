import numpy as np

gx = 0
gy = 0
gz = -9.81
m = 1
l = 2           # moment arm length

Ixx = 1
Iyy = 1
Izz = 1

g = np.array([
    gx,
    gy,
    gz
])

moment_arm = np.array([
    0,
    0,
    -l/2
])

I = np.array([
    [Ixx,0,0],
    [0,Iyy,0],
    [0,0,Izz]
])

I_diag = [Ixx, Iyy, Izz]

I_inv = np.linalg.inv(I)

