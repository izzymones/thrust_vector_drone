import do_mpc

# The simulator class increments the state during a simulation 
# and collects data for the run
def simulator(model, dt):

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = dt)
    simulator.setup()
    return simulator