import matplotlib.pyplot as plt
from matplotlib import rcParams
import do_mpc
import numpy as np
import sys
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import casadi as cas

from thruster import Thruster
from spacecraft import Spacecraft

import pickle

initial_orientation = np.array([0.0, 0.0, 0.0])
R0 = Rot.from_euler('xyz', initial_orientation)
q0 = R0.as_quat()
omega0 = np.zeros(3)
J = 1.0*np.eye(3)
C = np.eye(6)

thruster_positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [
                              0.0, 0.0, 1.0]])
thruster_positions = np.vstack((thruster_positions, thruster_positions))
thruster_positions = np.vstack((thruster_positions, 0.5*thruster_positions))
thruster_orientations = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
thruster_orientations = np.vstack((thruster_orientations, -thruster_orientations))
thruster_orientation = np.vstack((thruster_orientations, thruster_orientations))
thruster_thrusts = np.ones(thruster_positions.shape[0])

thrusters = [Thruster(pos, ori, thrust) for pos, ori, thrust in zip(
    thruster_positions, thruster_orientations, thruster_thrusts)]
spacecraft = Spacecraft(q0, np.zeros(3), J, thrusters, C)

model_type = 'continuous'
model = do_mpc.model.Model(model_type)

q = model.set_variable('_x',  'q', (3,1))
omega = model.set_variable('_x', 'omega', (3,1))
theta = model.set_variable('_x', 'theta', (len(thrusters),1))

t = model.set_variable('_u', 't', (len(thrusters),1))
T = spacecraft.T.copy()
u = T@t

q0 = cas.sqrt(1-cas.dot(q, q))

omega_true = (2/q0)*(q0**2*omega + q *
                     cas.dot(q, omega) - q0*cas.cross(q, omega))

Omega = J@omega_true
omega_tilde = 0.5*(q0*Omega + cas.cross(q, Omega))

omega_dot = np.linalg.inv(J)@(u-cas.cross(omega_true, Omega))
u_tilde = 0.5*(q0*omega_dot+cas.cross(q, omega_dot)) - \
    0.25*cas.dot(omega_true, omega_true)*q

theta_coeffs = 6*np.ones(len(thrusters))
d_theta = (theta_coeffs*t-4*theta**4)#/1500.0

model.set_rhs('q', omega)
model.set_rhs('omega', u_tilde)
model.set_rhs('theta', d_theta)

Q_q = 1e2*np.eye(3)
P_q = 1e5*np.eye(3)
Q_omega = 0*np.eye(3)
P_omega = 1e3*np.eye(3)
Q = np.block([[Q_q, np.zeros((3, 3))], [np.zeros((3, 3)), Q_omega]])
R = 100*np.eye(len(thrusters))

model.set_expression(expr_name='eta', expr=0.097*cas.log(t+0.005)+1.04)
model.set_expression(expr_name='low_pen', expr=1+100*(1-1/(1+cas.exp(-100*(t-0.2)))))
model.set_expression(expr_name='running_cost', expr=q.T @
                     Q_q@q + omega.T@Q_omega@omega +
                     (t*model.aux['low_pen']/model.aux['eta']).T@R@(t*model.aux['low_pen']/model.aux['eta']))
model.set_expression(expr_name='terminal_cost', expr=q.T @
                     P_q@q + omega.T@P_omega@omega)
model.set_expression(expr_name='omega_true', expr=omega_true)

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 25,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 0.04,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

lterm = model.aux['running_cost']
mterm = model.aux['terminal_cost']

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(t=0.00005*np.ones(len(thrusters)))

mpc.bounds['lower','_u','t'] = np.zeros(len(thrusters))
mpc.bounds['upper','_u','t'] = np.ones(len(thrusters))
mpc.bounds['upper', '_x', 'theta'] = np.ones(len(thrusters))
# mpc.terminal_bounds['lower', '_x', 'omega'] = -0.05*np.ones(3)
# mpc.terminal_bounds['upper', '_x', 'omega'] = 0.05*np.ones(3)

mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.04
}

simulator.set_param(**params_simulator)

simulator.setup()

# Initial state
x0 = np.array([0.4, -0.33, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

n_steps = 300
for k in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)


rcParams['axes.grid'] = True
rcParams['font.size'] = 12


fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16, 9))
graphics.plot_results()
graphics.reset_axes()
plt.show()

plt.ion()
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'small'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

from matplotlib.animation import FuncAnimation, FFMpegWriter
# The function describing the gif:
x_arr = mpc.data['_x']
u_arr = mpc.data['_u']

edge_lines = np.array([[[-1, -1, 0], [-1, 1, 0]], [[-1, -1, 0], [1, -1, 0]],
                   [[-1, 1, 0], [1, 1, 0]], [[1, -1, 0], [1, 1, 0]]], dtype=float)
rot1 = Rot.from_euler('xyz', [0.0, 90.0, 0.0], degrees=True)
back_edges = np.empty_like(edge_lines)
for i in range(edge_lines.shape[0]):
    for j in range(edge_lines.shape[1]):
        back_edges[i, j] = rot1.apply(edge_lines[i, j])
edge_lines_plus = edge_lines.copy()
edge_lines_minus = edge_lines.copy()
edge_lines_plus[:, :, 2] += 1
edge_lines_minus[:, :, 2] -= 1
back_edges_plus = back_edges.copy()
back_edges_minus = back_edges.copy()
back_edges_plus[:, :, 0] += 1
back_edges_minus[:, :, 0] -= 1
edge_lines = np.concatenate((edge_lines_plus, edge_lines_minus, back_edges_plus, back_edges_minus), axis=0)

def spaceship(x):
    x = x.flatten()
    q = x[:3]
    q = np.append(q, np.sqrt(1-np.sum(q**2)))
    r = Rot.from_quat(q)

    rotated_edges = np.empty_like(edge_lines)
    for i in range(edge_lines.shape[0]):
        for j in range(edge_lines.shape[1]):
            rotated_edges[i, j] = r.apply(edge_lines[i, j])
    rotated_edges = rotated_edges[:,:,:2]
    return rotated_edges


fig = plt.figure(figsize=(16, 9))

ax1 = plt.subplot2grid((5, 2), (0, 0), rowspan=5)
ax2 = plt.subplot2grid((5, 2), (0, 1))
ax3 = plt.subplot2grid((5, 2), (1, 1))
ax4 = plt.subplot2grid((5, 2), (2, 1))
ax5 = plt.subplot2grid((5, 2), (3, 1))
ax6 = plt.subplot2grid((5, 2), (4, 1))

ax2.set_ylabel('Quaternion Elements')
ax3.set_ylabel('$\\omega$ [rad/s]')
ax4.set_ylabel('$\\theta$')
ax5.set_ylabel('Input controls')
ax6.set_ylabel('Running cost')

# Axis on the right.
for ax in [ax2, ax3, ax4, ax5, ax6]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if ax != ax5:
        ax.xaxis.set_ticklabels([])

ax5.set_xlabel('time [s]')

mpc_graphics.add_line(var_type='_x', var_name='q', axis=ax2)
mpc_graphics.add_line(var_type='_aux', var_name='omega_true', axis=ax3)
mpc_graphics.add_line(var_type='_x', var_name='theta', axis=ax4)
mpc_graphics.add_line(var_type='_u', var_name='t', axis=ax5)
mpc_graphics.add_line(var_type='_aux', var_name='running_cost', axis=ax6)

ax1.axhline(0, color='black')

edges = []
for i in range(edge_lines.shape[0]):
    edges.append(ax1.plot([], [], '-o', linewidth=5, markersize=10))

ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()

def update(t_ind):
    lines= spaceship(x_arr[t_ind])
    for i in range(len(edges)):
        edges[i][0].set_data(lines[i][:,0], lines[i][:,1])
    mpc_graphics.plot_results(t_ind)
    # mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()


anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
gif_writer = FFMpegWriter(fps=20)
anim.save('anim_dip.mp4', writer=gif_writer)
plt.close()

with open('slew_data.pickle', 'wb') as f:
    pickle.dump(mpc.data, f, protocol=pickle.HIGHEST_PROTOCOL)

q = mpc.data['_x'][:, :3]
q_traj = np.append(q, np.sqrt(1-q**2), axis=1)

with open('slew_data_q.npz', 'wb') as f:
    np.save(f, q_traj)
