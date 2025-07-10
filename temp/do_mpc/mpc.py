import do_mpc
import casadi as ca
import numpy as np



def MPC(model, dt, silence_solver=False):

    mpc = do_mpc.controller.MPC(model)

    # not sure what all of these do yet
    mpc.settings.n_horizon = 10
    mpc.settings.n_robust = 1
    mpc.settings.open_loop = 0
    mpc.settings.t_step = dt
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.collocation_type = 'radau'
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 1
    mpc.settings.store_full_solution = True
    
    if silence_solver:
        mpc.settings.supress_ipopt_output()

    Q = ca.diag(12)

    # build up the cost function 
    xr = ca.vertcat(1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    x = ca.vertcat(model.x['x'], model.x['v'], model.x['q'], model.x['w'])
    error = x - xr

    lterm = error.T @ Q @ error

    mpc.set_objective(lterm=lterm, mterm=lterm) #

    mpc.set_rterm(u=np.array([1, 1, 1, 1], dtype=float))  # or  np.ones(4)
    mpc.setup()
    return mpc
