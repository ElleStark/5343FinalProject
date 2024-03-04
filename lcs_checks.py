"""
Script to check results of calculating dictionary of LCS lines across multiple points in time and space.
Uses results from flowfield.compute_ftle function, as demonstrated in DiscreteTurbulent_main.py.
Elle Stark February 2024
"""
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle

# Obtain LCS lines dictionary from pickle file
fname = 'data/LCS_data/alcs_backwardT0.6_spacing0.00025_0.6to10s_x200to800_y223to623.pickle'
with open(fname, 'rb') as handle:
    lcs_dict = pickle.load(handle)

min_frame = 0
max_frame = 201
min_x_index = 200
max_x_index = 800
min_y_index = 223
max_y_index = 623
# Obtain odor information to plot below LCS
with h5py.File('D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5', 'r') as f:
    # Odor data - select from odor pairs (cxa, cxb), where x is integers [1,...,8]
    # location of c1a = [0,0.00375]
    odor_a = f.get('Odor Data/c1a')[min_frame:max_frame, min_x_index:max_x_index, min_y_index:max_y_index].T

    # Numeric grids
    xmesh = f.get('Model Metadata/xGrid')[min_x_index:max_x_index, min_y_index:max_y_index].T
    ymesh = f.get('Model Metadata/yGrid')[min_x_index:max_x_index, min_y_index:max_y_index].T

# Plot LCS lines at a few times to check data
all_times = np.array(list(lcs_dict.keys()))
# select_times = [all_times[0], all_times[int(len(all_times)/4)], all_times[int(len(all_times)/2)],
#                 all_times[int(3*len(all_times)/4)], all_times[-1]]
# min_time_idx = 100
# max_time_idx = 120
# idxs = slice(min_time_idx, max_time_idx)

idxs = list(np.linspace(50, 200, 7).astype(int))
idxs = [30]
select_times = all_times[idxs]
select_odor = odor_a[:, :, idxs]

# Loop through selected times and create line plot of all LCS lines for that time
t_idx = 0
for time in select_times:
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.pcolormesh(xmesh, ymesh, select_odor[:, :, t_idx], cmap=plt.cm.Reds, vmin=0, vmax=0.25)
    plt.colorbar()
    for i in range(len(lcs_dict[time][0])):
        ax.plot(lcs_dict[time][0][i], lcs_dict[time][1][i], c='b',
                linewidth=0.5, linestyle="dashed")
    plt.xlim(0.15, 0.35)
    plt.ylim(-0.05, 0.05)
    ax.set_aspect('equal')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$y$ (m)')
    plt.title(f'Odor (red) & LCS (blue lines) at t={time} s \nc1a, T=-0.6, LCSstep=0.01m, eigmindist=0.015m')
    plt.savefig(f'plots/LCS_lines/c1a_subset_t{time}.png')
    plt.show()

    t_idx += 1



