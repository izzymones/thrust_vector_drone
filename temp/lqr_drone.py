
import casadi as ca
from casadi import sin, cos
import model_constants as mc
import numpy as np

import sys
sys.path.append('..')
from control_lib.model import LQRDModel
from control_lib.simulator import Simulator
from control_lib.plotter import Plotter2d, DronePlotter3d


p = ca.MX.sym('p', 3)
v = ca.MX.sym('v', 3)
q = ca.MX.sym('q', 3)
w = ca.MX.sym('w', 3)

state_names = ['$p_x$','$p_y$','$p_z$','$v_x$','$v_y$','$v_z$','$q_x$','$q_y$','$q_z$','$w_x$','$w_y$','$w_z$']
theta1 = ca.MX.sym('theta1')
theta2 = ca.MX.sym('theta2')
avg_thrust = ca.MX.sym('avg_thrust')
diff_thrust = ca.MX.sym('diff_thrust')

x = ca.vertcat(
    p,
    v,
    q,
    w,
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

x_fixed = [5,0,0,0,0,0,0,0,0,0,0,0]
u_fixed = [0,0,-mc.m*mc.g[2],0]
Q = np.eye(12)
R = np.eye(4)
dt = 0.01
t = 10

lqrdDrone = LQRDModel(x, 
                dx, 
                u, 
                constants, 
                constants_0, 
                dt,
                name='TVC Drone LQRD',
                state_names=state_names)

lqrdDrone.set_up_K(Q, R, x_fixed, u_fixed)

simulator = Simulator(lqrdDrone, x0, u0, t)
data = simulator.run()
# plotter = Plotter2d(lqrdDrone, data, t)
# plotter.plot()
plotter2 = DronePlotter3d(lqrdDrone, data, t)
plotter2.plot()