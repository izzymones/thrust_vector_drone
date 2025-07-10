import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from do_mpc import graphics, estimator
from casadi import sin, cos
import mpc_model_constants as mc
import do_mpc

import sys
sys.path.append('../..')

from control_lib.model import MPCModel
from simulator import simulator


show_animation = True
store_results = False

px = ca.MX.sym('px', 1)
py = ca.MX.sym('py', 1)
pz = ca.MX.sym('pz', 1)
vx = ca.MX.sym('vx', 1)
vy = ca.MX.sym('vy', 1)
vz = ca.MX.sym('vz', 1)
qx = ca.MX.sym('qx', 1)
qy = ca.MX.sym('qy', 1)
qz = ca.MX.sym('qz', 1)
wx = ca.MX.sym('wx', 1)
wy = ca.MX.sym('wy', 1)
wz = ca.MX.sym('wz', 1)

state_names = ['$p_x$','$p_y$','$p_z$','$v_x$','$v_y$','$v_z$','$q_x$','$q_y$','$q_z$','$w_x$','$w_y$','$w_z$']
theta1 = ca.MX.sym('theta1')
theta2 = ca.MX.sym('theta2')
avg_thrust = ca.MX.sym('avg_thrust')
diff_thrust = ca.MX.sym('diff_thrust')

p = ca.vertcat(
    px,
    py,
    pz
)
v = ca.vertcat(
    vx,
    vy,
    vz
)
q = ca.vertcat(
    qx,
    qy,
    qz
)
w = ca.vertcat(
    wx,
    wy,
    wz
)

x = ca.vertcat(
    p,
    v,
    q,
    w
)


u = ca.vertcat(
    theta1,
    theta2,
    avg_thrust,
    diff_thrust,
)

m = ca.MX.sym('m') 
l = ca.MX.sym('l') 
moment_arm = ca.MX.sym('l', 3) 
g = ca.MX.sym('g', 3) 
I_diag = ca.MX.sym('I', 3)


F = avg_thrust * ca.vertcat(
    sin(theta2),
    -sin(theta1)*cos(theta2),
    cos(theta1)*cos(theta2)
)

roll_moment = ca.vertcat(0, 0, diff_thrust)
M = ca.cross(moment_arm, F) + roll_moment

angular_momentum = ca.diag(I_diag) @ w

qw = (1 - (x[6])**2 - (x[7])**2 - (x[8])**2)**(0.5)


r_b2w = ca.vertcat(
    ca.horzcat(1 - 2*(x[7]**2 + x[8]**2), 2*(x[6]*x[7] - x[8]*qw), 2*(x[6]*x[8] + x[7]*qw)),
    ca.horzcat(2*(x[6]*x[7] + x[8]*qw), 1 - 2*(x[6]**2 + x[8]**2), 2*(x[7]*x[8] - x[6]*qw)),
    ca.horzcat(2*(x[6]*x[8] - x[7]*qw), 2*(x[7]*x[8] + x[6]*qw), 1 - 2*(x[6]**2 + x[7]**2)),

)

Q_omega = ca.vertcat(
    ca.horzcat(0, x[11], -x[10], x[9]),
    ca.horzcat(-x[11], 0, x[9], x[10]),
    ca.horzcat(x[10], -x[9], 0, x[11]),
    ca.horzcat(x[9], x[10], -x[11], 0)
)

q_full = ca.vertcat(x[6:9], qw)

dx = ca.vertcat(
    v,
    (r_b2w @ F) / m + g,
    0.5 * Q_omega[0:3, :] @ q_full,
    ca.solve(ca.diag(I_diag), M - ca.cross(w, angular_momentum))
)

constants = ca.vertcat(m, l, moment_arm, g, I_diag)

f = ca.Function(
    'f',
    [x, u, constants],
    [dx],
    ['x', 'u', 'constants'],
    ['dx']
)

f_jacobian = f.jacobian()

constants_0 = ca.DM([
    mc.m,
    mc.l,
    *mc.moment_arm,
    *mc.g,
    *mc.I_diag
])  

x0 = np.array([0,0,0,0,0,0,0,0,0,0,0.5,0])
u0 = np.array([0,0,0,0])

x_fixed = [0,0,0,0,0,0,0,0,0,0,0,0]
u_fixed = [0,0,-mc.m*mc.g[2],0]
Q = np.eye(12)
R = np.eye(4)
dt = 0.01
t = 10

MPCDrone = MPCModel(x, 
                dx, 
                u, 
                constants, 
                constants_0, 
                dt,
                name='TVC Drone MPC',
                state_names=state_names)

MPCDrone.set_up_K(Q, R, x_fixed, u_fixed)

mpc = MPCDrone.mpc
sim = simulator(MPCDrone.model, dt)
estimator = estimator.StateFeedback(MPCDrone.model)


# initial state
x0 = ca.vertcat(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
mpc.x0 = x0
sim.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()


fig, ax, graphics = graphics.default_plot(mpc.data)
plt.ion()


for k in range(200):
    u0 = mpc.make_step(x0)
    y_next = sim.make_step(u0)
    x0 = estimator.make_step(y_next)
    print('u', u0)
    print('state', x0)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)


