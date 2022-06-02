import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import pickle
import do_mpc

N = 100
with open('angular_data1.pickle', 'rb') as f:
    data = pickle.load(f)

with open('angular_data1_q.npz', 'rb') as f:
    q_traj = np.load(f)[:-1][:N]

ts = data['_time'][:N]
x = data['_x'][:N]
u = data['_u'][:N]
aux = data['_aux'][:N]

omega = aux[:, -3:][:N]
theta = x[:, -6:][:N]
control_profile = u

running_cost = data['_aux', 'running_cost'][:N]
terminal_cost = data['_aux', 'terminal_cost'][:N]

plt.style.use('seaborn-whitegrid')

fig, axs = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
ax = axs[1]
for i in range(3, 6):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, omega[:, i-3], label=f'$\\omega_{i-2}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=14)

ax = axs[0]
ax.plot(ts, q_traj[:, 3], label='$q^0$', color='k')
for i in range(0, 3):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, q_traj[:, i], label=f'$q_{i+1}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$q$', fontsize=14)
ax.set_title(
    'Model Predictive Control, Detumbling Maneuver', fontsize=17)

ax = axs[2]
for i in range(u.shape[1]):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, theta[:, i], label=f'$\\theta_{i+1}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$\\theta$', fontsize=14)
ax.set_ylim([-0.1, 1.1])

ax = axs[3]
for i in range(u.shape[1]):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, control_profile[:, i], label=f'$u_{i+1}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$v(t)$', fontsize=14)
ax.set_title('Control Inputs', fontsize=17)

plt.tight_layout()

plt.figure(figsize=(5, 5))
plt.plot(ts, running_cost+terminal_cost)
plt.xlabel('Time [s]')
plt.ylabel('$J^*(t)$')
plt.title('Detumbling Maneuver Cost Function')
plt.show()

with open('slew_data.pickle', 'rb') as f:
    data = pickle.load(f)

N = 100

with open('slew_data_q.npz', 'rb') as f:
    q_traj = np.load(f)[:,:3][:N]

q0_traj = np.sqrt(1-np.sum(q_traj**2, axis=1))
# q_traj = np.append(q_traj, np.sqrt(1-np.sum(q_traj**2, axis=1)), axis=0)

ts = data['_time'][:N]
x = data['_x'][:N]
u = data['_u'][:N]
aux = data['_aux'][:N]

omega = aux[:, -3:]
theta = x[:, -6:]
control_profile = u

running_cost = data['_aux', 'running_cost']
terminal_cost = data['_aux', 'terminal_cost']

fig, axs = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
ax = axs[1]
for i in range(3, 6):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, omega[:, i-3], label=f'$\\omega_{i-2}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=14)

ax = axs[0]
# ax.plot(ts, q0_traj, label='$q^0$', color='k')
for i in range(0, 3):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, q_traj[:, i], label=f'$q_{i+1}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$q$', fontsize=14)
ax.set_title(
    'Model Predictive Control, Slew Maneuver', fontsize=17)

ax = axs[2]
for i in range(u.shape[1]):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, theta[:, i], label=f'$\\theta_{i+1}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$\\theta$', fontsize=14)
ax.set_ylim([-0.1, 1.1])

ax = axs[3]
for i in range(u.shape[1]):
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(
        ts, control_profile[:, i], label=f'$u_{i+1}$', color=color)
ax.legend(loc='upper right', fontsize=14)
ax.set_xlabel('Time [s]', fontsize=14)
ax.set_ylabel('$v(t)$', fontsize=14)
ax.set_title('Control Inputs', fontsize=17)

plt.tight_layout()

plt.figure(figsize=(5, 5))
plt.plot(data['_time'], running_cost+terminal_cost)
plt.xlabel('Time [s]')
plt.ylabel('$J^*(t)$')
plt.title('Slew Maneuver Cost Function')
plt.show()


mpc_graphics = do_mpc.graphics.Graphics(data)

# The function describing the gif:
x_arr = data['_x']
u_arr = data['_u']

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
edge_lines = np.concatenate(
    (edge_lines_plus, edge_lines_minus, back_edges_plus, back_edges_minus), axis=0)


def spaceship(x):
    x = x.flatten()
    q = x[:3]
    q = np.append(q, np.sqrt(1-np.sum(q**2)))
    r = Rot.from_quat(q)

    rotated_edges = np.empty_like(edge_lines)
    for i in range(edge_lines.shape[0]):
        for j in range(edge_lines.shape[1]):
            rotated_edges[i, j] = r.apply(edge_lines[i, j])
    rotated_edges = rotated_edges[:, :, :2]
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
    lines = spaceship(x_arr[t_ind])
    for i in range(len(edges)):
        edges[i][0].set_data(lines[i][:, 0], lines[i][:, 1])
    mpc_graphics.plot_results(t_ind)
    # mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()


anim = FuncAnimation(fig, update, frames=200, repeat=False)
gif_writer = FFMpegWriter(fps=20)
anim.save('anim_dip.mp4', writer=gif_writer)
plt.close()