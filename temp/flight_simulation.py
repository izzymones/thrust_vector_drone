from basic_lqr.lqr_model import equations_of_motion, SSTVCModel
from time import perf_counter
import numpy as np
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
    print(-md.K)
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

    
def compare_solution_methods(method1, method2, time_series, x0, xr,):

    x_1 = method1.simulate(time_series, x0, xr)[0]
    x_2 = method2.simulate(time_series, x0, xr)[0]

    u_1 = method1.simulate(time_series, x0, xr)[1]
    u_2 = method2.simulate(time_series, x0, xr)[1]

    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({
    "text.usetex": True,
    })

    plt.plot(time_series,x_1[:, 0], label='x position 1', color='pink')
    plt.plot(time_series,x_1[:, 1], label='y position 1', color='red')
    plt.plot(time_series,x_1[:, 2], label='z position 1', color='orange')
    plt.plot(time_series,x_1[:, 3], label='x velocity 1', color='yellow')
    plt.plot(time_series,x_1[:, 4], label='y velocity 1', color='olive')
    plt.plot(time_series,x_1[:, 5], label='z velocity 1', color='green')
    # plt.plot(time_series,x_1[:, 6], label='x quaternion 1', color='cyan')
    # plt.plot(time_series,x_1[:, 7], label='y quaternion 1', color='blue')
    # plt.plot(time_series,x_1[:, 8], label='z quaternion 1', color='purple')
    # plt.plot(time_series,x_1[:, 9], label='x angular velocity 1', color='magenta')
    # plt.plot(time_series,x_1[:, 10], label='y angular velocity 1', color='brown')
    # plt.plot(time_series,x_1[:, 11], label='z angular velocity 1', color='black')

    plt.plot(time_series,u_1[:, 0], label='theta1 1', color='black')
    plt.plot(time_series,u_1[:, 1], label='theta2 1', color='brown')
    plt.plot(time_series,u_1[:, 2], label='avg_thrust 1', color='magenta')
    plt.plot(time_series,u_1[:, 3], label='diff_thrust 1', color='purple')


    # plt.plot(time_series,x_2[:, 0], label='x position 2', color='black')
    # plt.plot(time_series,x_2[:, 1], label='y position 2', color='brown')
    # plt.plot(time_series,x_2[:, 2], label='z position 2', color='magenta')
    # plt.plot(time_series,x_2[:, 3], label='x velocity 2', color='purple')
    # plt.plot(time_series,x_2[:, 4], label='y velocity 2', color='blue')
    # plt.plot(time_series,x_2[:, 5], label='z velocity 2', color='cyan')
    # plt.plot(time_series,x_2[:, 6], label='x quaternion 2', color='green')
    # plt.plot(time_series,x_2[:, 7], label='y quaternion 2', color='olive')
    # plt.plot(time_series,x_2[:, 8], label='z quaternion 2', color='yellow')
    # plt.plot(time_series,x_2[:, 9], label='x angular velocity 2', color='orange')
    # plt.plot(time_series,x_2[:, 10], label='y angular velocity 2', color='red')
    # plt.plot(time_series,x_2[:, 11], label='z angular velocity 2', color='pink')

    # plt.plot(time_series,u_2[:, 0], label='theta1 2', color='pink')
    # plt.plot(time_series,u_2[:, 1], label='theta2 2', color='red')
    # plt.plot(time_series,u_2[:, 2], label='avg_thrust 2', color='orange')
    # plt.plot(time_series,u_2[:, 3], label='diff_thrust 2', color='yellow')

    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend(loc='lower left')
    plt.legend()
    plt.show()

    

def run_comparison():
    tspan = np.arange(0,20,0.1)
    x0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0]) # Initial condition
    xr = np.array([0,2,0,0,0,0,0,1,1,0,0,0])      # Reference position 

    method1 = Method(ode)
    method2 = Method(lss)

    compare_solution_methods(method1, method2, tspan, x0, xr)

run_comparison()
