import numpy as np
import casadi as ca
from casadi import sin, cos
import mpc_model_constants as mc
import do_mpc

def model(symvar_type='SX'):

    # Initialize the model
    model_type = 'continuous' 
    model = do_mpc.model.Model(model_type, symvar_type)

    # state variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(3,1))
    v = model.set_variable(var_type='_x', var_name='v', shape=(3,1))
    q = model.set_variable(var_type='_x', var_name='q', shape=(3,1))
    w = model.set_variable(var_type='_x', var_name='w', shape=(3,1))

    state = ca.vertcat(x,v,q,w)


    # Input
    u = model.set_variable(var_type='_u', var_name='u', shape=(4,1))

    # defining the equations of motion for the model
    # we begin with some partial expressions to make the formulas easier to build

    F = u[2] * ca.vertcat(
        sin(u[1]),
        -sin(u[0])*cos(u[1]),
        cos(u[0])*cos(u[1])
    )

    roll_moment = ca.vertcat(0, 0, u[3])
    M = ca.cross(mc.moment_arm, F) + roll_moment

    angular_momentum = ca.diag(mc.I_diag) @ w

    qw = (1 - (state[6])**2 - (state[7])**2 - (state[8])**2)**(0.5)

    r_b2w = ca.vertcat(
        ca.horzcat(1 - 2*(state[7]**2 + state[8]**2), 2*(state[6]*state[7] - state[8]*qw), 2*(state[6]*state[8] + state[7]*qw)),
        ca.horzcat(2*(state[6]*state[7] + state[8]*qw), 1 - 2*(state[6]**2 + state[8]**2), 2*(state[7]*state[8] - state[6]*qw)),
        ca.horzcat(2*(state[6]*state[8] - state[7]*qw), 2*(state[7]*state[8] + state[6]*qw), 1 - 2*(state[6]**2 + state[7]**2)),
    )

    Q_omega = ca.vertcat(
        ca.horzcat(0, state[11], -state[10], state[9]),
        ca.horzcat(-state[11], 0, state[9], state[10]),
        ca.horzcat(state[10], -state[9], 0, state[11]),
        ca.horzcat(state[9], state[10], -state[11], 0)
    )


    q_full = ca.vertcat(state[6:9], qw)

    model.set_rhs('x', v)
    model.set_rhs('v', (r_b2w @ F) / mc.m + mc.g)
    model.set_rhs('q', 0.5 * Q_omega[0:3, :] @ q_full)
    model.set_rhs('w', ca.solve(ca.diag(mc.I_diag), M - ca.cross(w, angular_momentum)))


    model.setup()

    # this returns both the non-linear model and the linear model
    return model