import numpy as np

gx = 0
gy = 0
gz = -9.81
m = 1
l = 2           # moment arm length

Ixx = 0.06033711
Iyy = 0.061401623
Izz = 0.011873499
Ixy = 0.00004628
Ixz = 0.001382042
Iyz = 0.001000673

COM = np.array([
    0.005376,
    0.004033,
    0.203302
])

COT =  np.array([
    0.005376,
    0.0,
    0.016143
])


g = np.array([
    gx,
    gy,
    gz
])

moment_arm = COT - COM

I = np.array([
    [Ixx,Ixy,Ixz],
    [Ixy,Iyy,Iyz],
    [Ixz,Iyz,Izz]
])

I_diag = [Ixx, Iyy, Izz]

I_inv = np.linalg.inv(I)

