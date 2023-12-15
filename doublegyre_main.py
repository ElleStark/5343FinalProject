"""
Elle Stark Fall 2023
Replicate figures from Pratt et al., 2015, available at https://doi.org/10.1063/1.4914467

Project organization inspired by C++ implementation for LCS analysis found here: https://github.com/stevenliuyi/lcs
Code inspired by Python FTLE calculations used by: 1) Liu et al., 2018 (https://doi.org/10.1002/2017JC013390):
https://github.com/stevenliuyi/ocean-ftle and 2) https://github.com/jollybao/LCS
"""
import matplotlib.pyplot as plt
import numpy as np
import flowfield
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time




# Constants for double gyre:
a = 0.1  # velocity magnitude A aka U in Pratt et al., 2015
eps = 0.25
T_0 = 10

# Create double gyre object and calculate velocity fields
n = 1000  # number of grid steps in the x direction, use fewer when plotting velocity arrows
DoubleGyre = flowfield.DoubleGyre(a, eps, T_0, n)


# Check: Plot velocity field at a few times
# t = np.linspace(0, 3.5*T_0, 71, endpoint=True)  # use 3.5T_0 as max t to match Pratt et al., 2015
# DoubleGyre.compute_vfields(t)
# plot_times = [0, 0.25, 0.75]
# for time in plot_times:
#    plt.quiver(*DoubleGyre.velocity_fields[time*T_0])
#    plt.show()


# FTLE FIELD CALCULATIONS & MOVIE

# Find flow map using Runge-Kutta 4th order method to integrate backwards from t = t0 to t = t0-T
T = -2*T_0  # integration time - Pratt et al. used 2-2.5 turnover times
tau_list = np.linspace(0, T_0, 100)  # evolution times
#start_time = time.time()
#DoubleGyre.compute_flow_map(T, tau_list, dt=T/50, method='IE')
#print('time to compute trajectories is: ' + str(time.time()-start_time))

# If only interested in one or a few points in time, can use method 'compute_flow_map_w_trajs' and plot trajectories
# tau_list = [0]
# DoubleGyre.compute_flow_map_w_trajs(T, tau_list, dt=T/50, method='RK4')
# DoubleGyre.plot_trajectories([0, 2], [0, 1])

# Compute FTLE using central differencing for strain tensor
#start_time = time.time()
#DoubleGyre.compute_ftle()
#print('time to compute FTLE is: ' + str(time.time()-start_time))

# Create movie of FTLE field, passing in xlim and ylim for plotting. Saves ftle.mp4 in \plots\ folder.
#DoubleGyre.ftle_movie((0, 2), (0, 1))

# Plot FTLE field at a single point in time. Saves ftle_snap.png in \plots\ folder.
# DoubleGyre.ftle_snapshot(tau_list[0])

# Save individual FTLE field data if desired
# np.savetxt('data/doublegyre_negftle_t2.5T_T2T0_pratt.txt', DoubleGyre.ftle[tau_list[1]])


# PARTICLE TRACKING MODEL & MOVIE

num_particles = 10000  # number of particles initialized in each location
M = 1 / num_particles  # total mass = 1, so tag each particle with a mass of 1/#particles
dt = -T/500
D = 10 ** (-5)
blob1_ctr = [1.6, 0.5]  # red dye
blob2_ctr = [0.5, 0.5]  # blue dye

blob1_pos, blob2_pos = DoubleGyre.track_particles_rw(num_particles, blob1_ctr, blob2_ctr, dt, 4 * T_0, D)

# Create movies of particle tracking and FTLEs
plt.close('all')
fig, ax = plt.subplots()

ax.set(xlim=(0, 2), ylim=(0, 1))
ax.set_aspect('equal', adjustable='box')

# QC: plot first snapshot
# blobs_plot = ax.pcolormesh(blob1_pos[0, 0, :], blob1_pos[1, 0, :], blob1_pos[0], cmap=plt.cm.Reds)
# blobs_plot = ax.scatter(blob1_pos[:, 0, :])
ax.scatter(blob1_pos[0, 0, :], blob1_pos[1, 0, :], color='red')
ax.scatter(blob2_pos[0, 0, :], blob1_pos[1, 0, :], color='blue')
plt.show()

# def update(frame):
#     #for c in ax.collections:
#     #    c.remove()
#     #ax.pcolormesh(blob1_pos[0, frame, :], blob1_pos[1, frame, :], blob1_pos[frame], 100, cmap=plt.cm.Reds)
#     blobs_plot.set_offsets(blob1_pos[:, frame, :])
#
#     return blobs_plot
#
#
# blobs_movie = animation.FuncAnimation(fig=fig, func=update, frames=len(blob1_pos[0, :, 0]), interval=200)
#
# # save video
# f = r"plots/blobs_test.mp4"
# writervideo = animation.FFMpegWriter(fps=60)
# blobs_movie.save(f, writer=writervideo)



