from vpython import canvas, curve, vec, vector, color, label, arrow, cylinder, cone, rate
import numpy as np
from time import perf_counter
from math import sqrt

from model import equations_of_motion, SSTVCModel
from model_constants import *

# ------------------------------------------------------------------
# constants / initial conditions
tspan = np.arange(0, 10, 0.02)
x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2, 0, 0])
xr = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# ------------------------------------------------------------------
# dynamics helpers
def ode(tspan, x0, xr):
    x_data = np.empty([len(tspan), 12])
    x_data[0] = x0
    u_ref = np.array([0.0, 0.0, -m * g[2], 0.0])
    u_data = np.empty([len(tspan), 4])
    u_data[0] = u_ref
    u = u_ref
    x = x0
    dt = tspan[1] - tspan[0]
    start_time = perf_counter()
    for i in range(len(tspan)):
        dx = equations_of_motion(x, u)
        x = x + dx * dt
        u = np.asarray([0, 0, m * -g[2], 0] + -md.K @ (x - xr)).flatten()
        x_data[i] = x
        u_data[i] = u
    print('Time for', len(tspan), 'iterations of it_ode_sim: ', perf_counter() - start_time)
    return x_data, u_data

class Method:
    def __init__(self, simulation_function):
        self.method = simulation_function
    def simulate(self, time_series, x0, xr):
        return self.method(time_series, x0, xr)

md = SSTVCModel()
method = Method(ode)
x_hist, u_hist = method.simulate(tspan, x0, xr)

# ------------------------------------------------------------------
# quaternion tools
def quat_to_rot(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

def apply_pose(pos_model, q_vec):
    qw_sq = max(0.0, 1.0 - np.dot(q_vec, q_vec))
    qw = sqrt(qw_sq)
    quat = np.hstack((q_vec, qw))
    vp_pos = vector(pos_model[0], pos_model[2], -pos_model[1])
    body.pos = vp_pos
    R = quat_to_rot(quat)
    axis_m = R[:, 2]
    vp_axis = vector(axis_m[0], axis_m[2], -axis_m[1])
    body.axis = vp_axis * rocket_length
    nose.pos = body.pos + body.axis
    nose.axis = vp_axis * nose_length
    thrust.pos = body.pos
    thrust.axis = -vp_axis.norm() * thrust_length

# ------------------------------------------------------------------
# scene setup
scene = canvas(title="3D Rocket Visualization", background=color.white, width=1600, height=1200)
scene.forward = vector(-1, -1, -1)
scene.up = vector(0, 1, 0)
scene.range = 2.5

def draw_grid(half_extent=10, spacing=1, y_level=0, col=color.gray(0.9)):
    r = 0.002
    for x in range(-half_extent, half_extent + 1, spacing):
        curve(vec(x, y_level, -half_extent), vec(x, y_level, half_extent), color=col, radius=r)
    for z in range(-half_extent, half_extent + 1, spacing):
        curve(vec(-half_extent, y_level, z), vec(half_extent, y_level, z), color=col, radius=r)
draw_grid()

axis_len = 2
arrow(pos=vec(0, 0, 0), axis=vec(axis_len, 0, 0), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.red)
arrow(pos=vec(0, 0, 0), axis=vec(0, axis_len, 0), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.blue)
arrow(pos=vec(0, 0, 0), axis=vec(0, 0, -axis_len), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.green)
label(pos=vec(axis_len, 0, 0), text='X', box=False, opacity=0, color=color.red)
label(pos=vec(0, axis_len, 0), text='Z', box=False, opacity=0, color=color.blue)
label(pos=vec(0, 0, -axis_len), text='Y', box=False, opacity=0, color=color.green)

rocket_radius = 0.1
rocket_length = 1.0
nose_length = 0.2
thrust_length = 0.5

body = cylinder(pos=vec(0, 0.01, 0), axis=vec(0, rocket_length, 0), radius=rocket_radius, color=color.red)
nose = cone(pos=body.pos + body.axis, axis=vec(0, nose_length, 0), radius=rocket_radius, color=color.orange)
thrust = arrow(pos=body.pos, axis=vec(0, -thrust_length, 0), shaftwidth=0.05, headwidth=0.08, headlength=0.1, color=color.blue)

# ------------------------------------------------------------------
# animation loop
dt_vis = tspan[1] - tspan[0]
for k in range(len(tspan)):
    rate(1 / dt_vis)
    pos = x_hist[k, 0:3]
    qv = x_hist[k, 6:9]
    apply_pose(pos, qv)
    scene.center = body.pos + vector(0, 0.3, 0)