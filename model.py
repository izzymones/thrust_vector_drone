import numpy as np
from model_constants import *
from control.matlab import lqr
import matplotlib.pyplot as plt



def equations_of_motion(x,u):
    avg_thrust = u[2]
    diff_thrust = u[3]

    # these equalities are not accurate and 
    # the relationship must be found experimentally
    thrust_magnitude = avg_thrust
    roll_moment = np.array([0, 0, diff_thrust])

    F = np.array([
        thrust_magnitude*np.sin(u[1]),
        -thrust_magnitude*np.sin(u[0])*(np.cos(u[1])),
        thrust_magnitude*np.cos(u[0])*np.cos(u[1])
    ])


    M = np.cross(moment_arm, F) + roll_moment

    angular_momentum = I @ x[9:12]

    qw = (1 - (x[6])**2 - (x[7])**2 - (x[8])**2)**(0.5)

    r_b2w = np.array([
        [1 - 2*(x[7]**2 + x[8]**2), 2*(x[6]*x[7] - x[8]*qw), 2*(x[6]*x[8] + x[7]*qw)],
        [2*(x[6]*x[7] + x[8]*qw), 1 - 2*(x[6]**2 + x[8]**2), 2*(x[7]*x[8] - x[6]*qw)],
        [2*(x[6]*x[8] - x[7]*qw), 2*(x[7]*x[8] + x[6]*qw), 1 - 2*(x[6]**2 + x[7]**2)]
    ])

    Q_omega = np.array([
        [0, x[11], -x[10], x[9]],
        [-x[11], 0, x[9], x[10]],
        [x[10], -x[9], 0, x[11]],
        [x[9], x[10], -x[11], 0]
    ])

    dx = np.zeros(12)

    q = np.hstack((x[6:9], qw))

    dx[0:3] = x[3:6]
    dx[3:6] = (r_b2w @ F) / m + g
    dx[6:9] = 0.5 * (Q_omega[:3, :] @ q)
    dx[9:12] = I_inv @ (M - (np.cross(x[9:12], angular_momentum)))
    
    return dx

# State Space Model
class SSTVCModelConstants:
    # equations of motion linearized about hovering at the origin

    A = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 0, 0, -2*g[2], 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 0, 2*g[2], 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # linearization of control matrix

    B = np.array([
        [0,0,0,0],\
        [0,0,0,0],\
        [0,0,0,0],\
        [0,-g[2],0,0],\
        [g[2],0,0,0],\
        [0,0,1/m,0],\
        [0,0,0,0],\
        [0,0,0,0],\
        [0,0,0,0],\
        [(l*m*g[2])/Ixx,0,0,0],\
        [0,(l*m*g[2])/Iyy,0,0],\
        [0,0,0,1/Izz],\
    ])

    Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    R = np.diag([1, 1, 1, 1])
    K = lqr(A,B,Q,R)[0]

class SSTVCModel:
    def __init__(self):
        # Set constants from separate classes as attributes
        for cls in [SSTVCModelConstants]:
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    self.__dict__.update(**{key: value})

    # these are constants. This keeps them read only
    def __setattr__(self, name, value):
        raise TypeError("Model values are immutable")

if __name__ == "__main__":

    N = 100
    x = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    u = np.array([0,0,0,0])
    dt = 0.1

    x_history = np.zeros((N, 12))
    qw_history = np.zeros((N, 1))


    for i in range(N):
        dx = equations_of_motion(x,u)
        x = x + dt * dx

        qw = (1 - (x[6])**2 - (x[7])**2 - (x[8])**2)**(0.5)



        x_history[i, :] = x
        qw_history[i, :] = qw

    plt.xlabel("Time step")
    plt.ylabel("x")
    plt.title("x over time")
    plt.legend()
    plt.plot(x_history[:, 0], label='x position', color='red')
    plt.plot(x_history[:, 1], label='y position', color='orange')
    plt.plot(x_history[:, 2], label='z position', color='yellow')
    plt.plot(x_history[:, 3], label='x velocity', color='green')
    plt.plot(x_history[:, 4], label='y velocity', color='blue')
    plt.plot(x_history[:, 5], label='z velocity', color='purple')
    plt.plot(x_history[:, 6], label='qx', color='pink')
    plt.plot(x_history[:, 7], label='qy', color='olive')
    plt.plot(x_history[:, 8], label='qz', color='magenta')

    plt.plot(qw_history, label='qw', color='cyan')

    plt.plot(x_history[:, 9], label='angvel x', color='brown')
    plt.plot(x_history[:, 10], label='angvel y', color='gray')
    plt.plot(x_history[:, 11], label='angvel z', color='black')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()
