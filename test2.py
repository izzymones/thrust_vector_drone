from model import equations_of_motion, SSTVCModel
from time import perf_counter
import numpy as np
import plotly.graph_objects as go
from time import sleep
from model_constants import *
import matplotlib.pyplot as plt



md = SSTVCModel()


def ode(tspan, x0, xr):
    x_data = np.empty([len(tspan),12])
    x_data[0] = x0
    u_ref = np.array([0.0,0.0,-m*g[2],0.0])
    u_data = np.empty([len(tspan),4])
    u_data[0] = u_ref
    u = u_ref
    x = x0
    dt = tspan[1]-tspan[0]
    start_time = perf_counter()
    for i in range(len(tspan)):
        dx = equations_of_motion(x,u)
        x = x + dx*dt
        u = np.asarray([0, 0, m*-g[2], 0] + -md.K @ (x - xr)).flatten()
        x_data[i] = x
        u_data[i] = u


    print('Time for', len(tspan), 'iterations of it_ode_sim: ', perf_counter() - start_time)
    return x_data, u_data


def lss(tspan, x0, xr):
    x_data = np.empty([len(tspan),12])
    x_data[0] = x0
    u_ref = np.array([0.0,0.0,-m*g[2],0.0])
    u_data = np.empty([len(tspan),4])
    u_data[0] = u_ref
    x = x0
    dt = tspan[1]-tspan[0]
    start_time = perf_counter()
    for i in range(len(tspan)):
        du = np.asarray(-md.K @ (x - xr)).flatten()
        dx = md.A @ (x-xr) + md.B @ du
        x = x + dx*dt
        x_data[i] = x
        u_data[i] = u_ref + du

    print('Time for', len(tspan), 'iterations of it_ls_sim:', perf_counter() - start_time)
    return x_data, u_data


class Method:
    def __init__(self, simulation_function):
        self.method = simulation_function

    def simulate(self, time_series, x0, xr):
        return self.method(time_series, x0, xr)

def rot_matrix_from_vec3(e):
    qx, qy, qz = e
    qw = np.sqrt(max(0.0, 1.0 - qx*qx - qy*qy - qz*qz))
    return np.array([
        [1-2*(qy*qy+qz*qz),      2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [    2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [    2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])

def mesh_cylinder(p0, p1, R=0.15, res=24, color='lightblue'):
    p0, p1 = map(np.asarray, (p0, p1))
    v = p1-p0; h=np.linalg.norm(v); v=v/h
    notv = np.array([1,0,0]) if abs(v[0])<0.9 else np.array([0,1,0])
    n1=np.cross(v,notv); n1/=np.linalg.norm(n1)
    n2=np.cross(v,n1)
    ang=np.linspace(0,2*np.pi,res)
    circ=np.array([R*np.cos(ang),R*np.sin(ang)])
    bot=p0[:,None]+n1[:,None]*circ[0]+n2[:,None]*circ[1]
    top=bot+v[:,None]*h
    X=np.concatenate([bot[0],top[0]]); Y=np.concatenate([bot[1],top[1]]); Z=np.concatenate([bot[2],top[2]])
    tri=[]
    for i in range(res-1):
        tri+=[[i,i+1,i+res+1],[i,i+res+1,i+res]]
    tri+=[[res-1,0,res],[res-1,res,2*res-1]]
    i,j,k=zip(*tri)
    return go.Mesh3d(x=X,y=Y,z=Z,i=i,j=j,k=k,color=color,opacity=0.8)

def mesh_cone(base, dirvec, h=0.3, R=0.15, res=24, color='orange'):
    base=np.asarray(base); d=np.asarray(dirvec); d/=np.linalg.norm(d)
    apex=base+d*h
    notd=np.array([1,0,0]) if abs(d[0])<0.9 else np.array([0,1,0])
    n1=np.cross(d,notd); n1/=np.linalg.norm(n1)
    n2=np.cross(d,n1)
    ang=np.linspace(0,2*np.pi,res)
    circ=np.array([R*np.cos(ang),R*np.sin(ang)])
    ring=base[:,None]+n1[:,None]*circ[0]+n2[:,None]*circ[1]
    X=np.append(ring[0],apex[0]); Y=np.append(ring[1],apex[1]); Z=np.append(ring[2],apex[2])
    faces=[[i,i+1,res] for i in range(res-1)]+[[res-1,0,res]]
    i,j,k=zip(*faces)
    return go.Mesh3d(x=X,y=Y,z=Z,i=i,j=j,k=k,color=color,opacity=0.9)


def simulate_3d(method, time_series, x0, xr):

    # ------------------------------------------------------------------
    # 1. simulate (we visualise x_1, u_1 from method1 / ode)
    # ------------------------------------------------------------------
    x_1, u_1 = method.simulate(time_series, x0, xr)

    body_tip = np.array([0, 0, 1])         # 1-m body-Z axis in body frame
    body_ori = np.zeros(3)

    frames = []
    for k in range(len(time_series)):
        R   = rot_matrix_from_vec3(x_1[k, 6:9])   # body  → world
        pos = x_1[k, 0:3]                         # vehicle origin in world

        # blue cylinder endpoints
        p0 = pos                                 # base  (body origin)
        p1 = pos + R @ body_tip                  # tip   (1 m up in body-Z)

        # ---------- red force vector ----------------------------------
        theta1, theta2, Tbar = u_1[k, 0:3]       # [θ1, θ2, avg_thrust]

        force_body = np.array([
            Tbar * np.sin(theta2),
           -Tbar * np.sin(theta1) * np.cos(theta2),
            Tbar * np.cos(theta1) * np.cos(theta2)
        ])                                        # body coords

        force_world = R @ force_body
        norm_fw = np.linalg.norm(force_world)

        if norm_fw < 1e-6:                 # avoid div-by-zero
            force_unit = np.zeros(3)
        else:
            force_unit = force_world / norm_fw

        L = 0.25                           # visible half-length (m)
        f_tail = p0 - L * force_unit       # start  – below the base
        f_head = p0                        # end    – at the base

        # ---------- traces -------------------------------------------
        cyl  = mesh_cylinder(p0, p1)                    # blue
        cone = mesh_cone(p1, p1 - p0)                   # orange
        fvec = go.Scatter3d(                            # red force line
            x=[f_tail[0], f_head[0]],
            y=[f_tail[1], f_head[1]],
            z=[f_tail[2], f_head[2]],
            mode="lines",
            line=dict(color="red", width=6),
            showlegend=False
        )

        frames.append(go.Frame(data=[cyl, cone, fvec], name=str(k)))

    # ------------------------------------------------------------------
    # 2. figure & animation controls
    # ------------------------------------------------------------------
    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        uirevision="camera",           # keep user drag / zoom
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[-10, 10], autorange=False),
            yaxis=dict(range=[-10, 10], autorange=False),
            zaxis=dict(range=[-10, 10], autorange=False)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 40, "redraw": True},
                        "transition": {"duration": 0},
                        "fromcurrent": True
                    }]
                },
                {
                    "label": "❚❚ Pause",
                    "method": "animate",
                    "args": [[None], {"mode": "immediate"}]
                }
            ]
        }]
    )


    fig.show(renderer="browser")

def run_simulation():
    tspan = np.arange(0,5,0.05)
    x0 = np.array([0,0,0,0,0,0,0,0,0,0.5,0.5,0.5])
    xr = np.array([1,1,5,0,0,0,0,0,0,0,0,0])

    method = Method(ode)

    simulate_3d(method, tspan, x0, xr)

run_simulation()
