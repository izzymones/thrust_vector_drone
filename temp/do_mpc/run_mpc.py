import matplotlib.pyplot as plt
from do_mpc import graphics, estimator
from do_mpc.data import save_results
import casadi as ca

from dompcmodel import model
from mpc import MPC
from simulator import simulator


show_animation = True
store_results = False


dt = 0.1
model = model()
mpc = MPC(model, dt)
sim = simulator(model, dt)
estimator = estimator.StateFeedback(model)


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

input('Press any key to exit.')
# plot graphics
# fig, ax, graphics = graphics.default_plot(sim.data, figsize=(12,8))

# fig, ax, graphics = graphics.default_plot(mpc.data)
# graphics.plot_results()
# graphics.reset_axes()
# plt.savefig("mpc.pdf", format="pdf", bbox_inches="tight")
# plt.show()


# Store results:
if store_results:
    save_results([mpc, sim], 'two_wheel_balancing_bot_MPC')