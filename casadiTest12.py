
import casadi as ca
from casadi import sin, cos
import model_constants as mc
import numpy as np

p = ca.MX.sym('p', 3)
v = ca.MX.sym('v', 3)
q = ca.MX.sym('q', 3)
w = ca.MX.sym('w', 3)

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
l = ca.MX.sym('l', 3) 
g = ca.MX.sym('g', 3) 
I_diag = ca.MX.sym('I', 3)


F = avg_thrust * ca.vertcat(
    sin(theta2),
    -sin(theta1)*cos(theta2),
    cos(theta1)*cos(theta2)
)

roll_moment = ca.vertcat(0, 0, diff_thrust)
M = ca.cross(l, F) + roll_moment

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

constants = ca.vertcat(m, l, g, I_diag)

f = ca.Function(
    'f',
    [x, u, constants],
    [dx],
    ['x', 'u', 'constants'],
    ['dx']
)

f_jacobian = f.jacobian()
print('jacobian', f_jacobian)

constants_0 = ca.DM([
    mc.m,
    *mc.moment_arm,
    *mc.g,
    *mc.I_diag
])  
x_0 = [0,0,0,0,0,0,0,0,0,0,0,0]
u_0 = [0,0,-mc.m*mc.g[2],0]

print('x_0', x_0, 'u_0', u_0, 'constants', constants_0)
linearized = f_jacobian(x = x_0, u = u_0, constants = constants_0)
A = linearized['jac_dx_x'].full()
B = linearized['jac_dx_u'].full()

print(A)
print(B)