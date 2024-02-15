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
from matplotlib import colors
import time
import utils
import scipy.io


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
tau_list = np.linspace(0, 4*T_0, 1000)  # evolution times
#tau_list = [3.5*T_0]
dt = T/500
start_time = time.time()
DoubleGyre.compute_flow_map(T, tau_list, dt=T/50, method='IE')
print('time to compute trajectories is: ' + str(time.time()-start_time))

# If only interested in one or a few points in time, can use method 'compute_flow_map_w_trajs' and plot trajectories
# tau_list = [0]
# DoubleGyre.compute_flow_map_w_trajs(T, tau_list, dt=dt, method='RK4')
# DoubleGyre.plot_trajectories([0, 2], [0, 1])

# Compute FTLE using central differencing for strain tensor
start_time = time.time()
DoubleGyre.compute_ftle()
print('time to compute FTLE is: ' + str(time.time()-start_time))

# Save FTLE field for all times
np.save('data/dg_ftle_4T0_n1000_1000step', DoubleGyre.ftle)

# Create movie of FTLE field, passing in xlim and ylim for plotting. Saves ftle.mp4 in \plots\ folder.
#DoubleGyre.ftle_movie((0, 2), (0, 1))

# Plot FTLE field at a single point in time. Saves ftle_snap.png in \plots\ folder.
#DoubleGyre.ftle_snapshot(tau_list[0], name='5343test')

# Save individual FTLE field data if desired
# np.savetxt('data/doublegyre_negftle_t2.5T_T2T0_pratt.txt', DoubleGyre.ftle[tau_list[1]])


# PARTICLE TRACKING MODEL & MOVIE
num_particles = 3000000  # number of particles initialized in each location
M = 1 / num_particles  # total mass = 1, so tag each particle with a mass of 1/#particles
dt = abs(dt)
D = 10 ** (-6)
batchelor_length = 0.1  # for setting up size of square for initial conditions
blob1_ctr = [1.6, 0.5]  # red dye
blob2_ctr = [0.5, 0.5]  # blue dye

# Function call to create and save particle tracking model
starttime = time.time()
blob1_pos, blob2_pos = DoubleGyre.track_particles_rw(num_particles, blob1_ctr, blob2_ctr, dt,
                                                    4 * T_0, D, batchelor_length)
np.save('data/blob1_pos_3m_dt500', blob1_pos)
np.save('data/blob2_pos_3m_dt500', blob2_pos)
print('time to compute particle tracking is: ' + str(time.time()-starttime))

# # QC: scatterplot first snapshot
# #ax.pcolormesh(blob1_conc[0, :, :], cmap=plt.cm.Reds)
# blob1_plot = ax.scatter(blob1_pos[0, 0, :], blob1_pos[1, 0, :], color='red')
# blob2_plot = ax.scatter(blob2_pos[0, 0, :], blob2_pos[1, 0, :], color='blue')
# plt.show()

# Load particle tracking data if already computed and saved
blob1_pos = np.load('data/blob1_pos_3m_dt500.npy')
blob2_pos = np.load('data/blob2_pos_3m_dt500.npy')

# First 2D histogram
blob1_ic, xbins1, ybins1 = np.histogram2d(blob1_pos[0, 0, :], blob1_pos[1, 0, :],
                                            bins=(np.linspace(0, 2, 1000), np.linspace(0, 1, 500)))
blob2_ic, xbins2, ybins2 = np.histogram2d(blob2_pos[0, 0, :], blob2_pos[1, 0, :],
                                            bins=(np.linspace(0, 2, 1000), np.linspace(0, 1, 500)))

# # QC Plot: Initial Conditions
# blob1_ic = (blob1_ic - np.min(blob1_ic)) / (np.max(blob1_ic) - np.min(blob1_ic))
# blob2_ic = (blob2_ic - np.min(blob2_ic)) / (np.max(blob2_ic) - np.min(blob2_ic))
# combined_ic = utils.color_change_white(blob2_ic, blob1_ic, 1)
# plt.pcolormesh(combined_ic.transpose(1, 0, 2))
# plt.show()
# plt.close()

# Normalized data for initial conditions
blob1_data = (blob1_ic - np.min(blob1_ic)) / (np.max(blob1_ic) - np.min(blob1_ic))
blob2_data = (blob2_ic - np.min(blob2_ic)) / (np.max(blob2_ic) - np.min(blob2_ic))

# Convert particle count data to Red-Blue color data - scale of 0.5 chosen for aesthetics
combined_data = utils.color_change_white(blob2_data, blob1_data, 0.5)
rxn_data = np.multiply(blob2_ic, blob1_ic)

if np.max(rxn_data) > 0:
    rxn_data = (rxn_data - np.min(rxn_data)) / (np.max(rxn_data) - np.min(rxn_data))
    rxn_data = rxn_data ** (1 / 4)  # scale data for better plotting

# Define reaction colormap
rxn_colors = scipy.io.loadmat('data/rxn_cmap.mat')
rxn_colors = rxn_colors['rxn_cmap']
rxn_cmap = colors.ListedColormap(rxn_colors)

# Create movies of particle tracking and FTLEs
plt.close('all')
# create a figure with two subplots, top for particle tracking ('blobs_plot') and bottom for FTLE w/rxn ('ftle_plot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 4.6))

for ax in [ax1, ax2]:
    ax.set(xlim=(0, 2), ylim=(0, 1))
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

# First snapshot - particle tracking
blobs_plot = ax1.imshow(np.zeros((len(xbins1), len(ybins1), 3)), extent=[0, 2, 0, 1], origin='lower')
blobs_plot.set_data(combined_data.transpose(1, 0, 2))

# First snapshot - FTLE with reaction
rxn_plot = ax2.imshow(rxn_data.T, extent=[0, 2, 0, 1], alpha=0.55, origin='lower', cmap=rxn_cmap, zorder=1)
ftle_plot = ax2.contourf(DoubleGyre.x, DoubleGyre.y, DoubleGyre.ftle[tau_list[0]], 100, extent=[0, 2, 0, 1],
                         cmap=plt.cm.Greys, zorder=-1)

# # QC: Plot and save snapshot
plt.savefig('plots/plot750.png', dpi=200)
plt.show()


def update(frame):
    # PARTICLE TRACKING PLOT (TOP POSITION)
    # Obtain particle tracking data for this frame using 2D histogram
    blob1_data, xbins1, ybins1 = np.histogram2d(blob1_pos[0, frame, :], blob1_pos[1, frame, :],
                                            bins=(np.linspace(0, 2, 1000), np.linspace(0, 1, 500)))
    blob2_data, xbins2, ybins2 = np.histogram2d(blob2_pos[0, frame, :], blob2_pos[1, frame, :],
                                            bins=(np.linspace(0, 2, 1000), np.linspace(0, 1, 500)))
    # Normalize 0 to 1 for input to color mapping
    blob1_data = (blob1_data - np.min(blob1_data)) / (np.max(blob1_data) - np.min(blob1_data))
    blob2_data = (blob2_data - np.min(blob2_data)) / (np.max(blob2_data) - np.min(blob2_data))
    # Convert particle count data to Red-Blue color data - scale of 0.5 chosen for aesthetics
    combined_data = utils.color_change_white(blob2_data, blob1_data, 0.5)
    blobs_plot.set_array(combined_data.transpose(1, 0, 2))

    # FTLE PLOT (BOTTOM POSITION)

    # Remove previous FTLE contours to update contourf correctly
    for c in ax2.collections:
        c.remove()
    ftle_plot = ax2.contourf(DoubleGyre.x, DoubleGyre.y, DoubleGyre.ftle[tau_list[frame]], 100, extent=[0, 2, 0, 1],
                         cmap=plt.cm.Greys, zorder=-1)

    # Had to also remove previous images to avoid overlay issues with imshow
    for img in ax2.images:
        img.remove()
    rxn_data = np.multiply(blob2_data, blob1_data)
    if np.max(rxn_data) > 0:
        rxn_data = (rxn_data - np.min(rxn_data)) / (np.max(rxn_data) - np.min(rxn_data))
        rxn_data = rxn_data ** (1 / 4)  # scale data for better plotting

    rxn_plot = ax2.imshow(rxn_data.T, extent=[0, 2, 0, 1], alpha=0.55, origin='lower', cmap=rxn_cmap, zorder=1)

    return blobs_plot, *ftle_plot.collections, rxn_plot


doublegyre_movie = animation.FuncAnimation(fig=fig, func=update, frames=len(blob1_pos[0, :, 0]), interval=200)

# save video
f = r"plots/doublegyre_final.mp4"
writervideo = animation.FFMpegWriter(fps=60)
doublegyre_movie.save(f, writer=writervideo, dpi=200)



