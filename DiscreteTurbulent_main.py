"""
Elle Stark October 2023
Expand FTLE computation code to handle discrete velocity fields.
"""
import pickle

import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import flowfield
import math

# 1. OBTAIN DATA

# SUBSET: specify desired time and space limits for reading in only a subset of the data

# Subset in time: SET TO None IF ALL TIMES DESIRED!
min_frame = 0
max_frame = 201  # 250 frames is 5 seconds of data for 50Hz resolution
# Subset in Space: SET TO None if ENTIRE DOMAIN DESIRED!
min_x_index = None
max_x_index = None  # 201 steps is 0.1 m for dx=0.005
min_y_index = None  # 423 is approx index of the center of multisource plume data
max_y_index = None

# Read in all needed data from hdf5 files. For more info on hdf5 format, see https://www.hdfgroup.org/.
# Use 'with' context manager to make sure h5 file is closed properly after using.
# Use [] to store data in memory, and use .item() method to convert single values from array to float.

start = time.time()  # track amount of time to read in the data
with h5py.File('D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5', 'r') as f:

    # Metadata: spatiotemporal resolution and domain size
    dt_freq = f.get('Model Metadata/timeResolution')[0].item()
    dt_data = 1 / dt_freq  # convert from Hz to seconds
    time_array_data = f.get('Model Metadata/timeArray')[:]
    spatial_res = f.get('Model Metadata/spatialResolution')[0].item()
    domain_size = f.get('Model Metadata/domainSize')
    domain_width = domain_size[0].item()  # [m] cross-stream distance
    domain_length = domain_size[1].item()  # [m] stream-wise distance

    # Numeric grids
    xmesh_uv = f.get('Model Metadata/xGrid')[min_x_index:max_x_index, min_y_index:max_y_index].T
    ymesh_uv = f.get('Model Metadata/yGrid')[min_x_index:max_x_index, min_y_index:max_y_index].T

    # Velocities: for faster reading, can read in subset of u and v data here
    # dimensions of multisource plume data (time, columns, rows) = (3001, 1001, 846)
    u_data = f.get('Flow Data/u')[min_frame:max_frame, min_x_index:max_x_index, min_y_index:max_y_index].T
    v_data = f.get('Flow Data/v')[min_frame:max_frame, min_x_index:max_x_index, min_y_index:max_y_index].T

    # Odor data - select from odor pairs (cxa, cxb), where x is integers [1,...,8]
    # higher numbers indicate increasing distance from each other. Source locations are symmetric about y=0.
    # location of c1a = [0,0.00375], location of c1b = [0, -0.00375]
    odor_a = f.get('Odor Data/c1a')[min_frame:max_frame, min_x_index:max_x_index, min_y_index:max_y_index].T
    odor_b = f.get('Odor Data/c1b')[min_frame:max_frame, min_x_index:max_x_index, min_y_index:max_y_index].T

# Track and display how long it took to read in the data
total_time = time.time()-start
print('time to read in data: ' + str(total_time))

# QC Plots: udata and vdata for comparing to interpolated figure
# plt.pcolor(u_data[:, :, 25])
# plt.savefig('plots/u_data_t0_5.png')
# plt.show()
# plt.close()
# plt.pcolor(v_data[:, :, 25])
# plt.savefig('plots/v_data_t0_5.png')
# plt.show()
# plt.close()
# plt.pcolor(u_data[:, :, 50])
# plt.savefig('plots/u_data_t1.png')
# plt.show()
# plt.close()
# plt.pcolor(v_data[:, :, 50])
# plt.savefig('plots/v_data_t1.png')
# plt.show()
# plt.close()

# Create grid of particles with desired spacing
particle_spacing = spatial_res / 2  # can determine visually if dx is appropriate based on smooth contours for FTLE
# Test with very coarse grid
# particle_spacing = 0.002

# x and y vectors based on velocity mesh limits and particle spacing
xvec_ftle = np.linspace(xmesh_uv[0][0], xmesh_uv[0][-1], int(np.shape(u_data)[1] * spatial_res/particle_spacing))
yvec_ftle = np.linspace(ymesh_uv[0][0], ymesh_uv[-1][0], int(np.shape(u_data)[0] * spatial_res/particle_spacing))
xmesh_ftle, ymesh_ftle = np.meshgrid(xvec_ftle, yvec_ftle, indexing='xy')
ymesh_ftle = np.flipud(ymesh_ftle)

# QC plot: initial positions of Lagrangian tracers
# fig, ax = plt.subplots()
# plt.scatter(xmesh_ftle, ymesh_ftle)
# ax.set_aspect('equal', adjustable='box')
# plt.xlim(0, 0.01)
# plt.ylim(0, 0.01)
# plt.show()

# Create DiscreteFlow object from flowfield to make use of flow map and ftle computation methods
turb_lcs = flowfield.DiscreteFlow(xmesh_ftle, ymesh_ftle, u_data, v_data, xmesh_uv, ymesh_uv, dt_data)

# #QC Plot: check velocities at a few times
# t = [0, 0.5, 1]  # use 3.5T_0 as max t to match Pratt et al., 2015
# turb_lcs.compute_vfields(t)
#
# for time in t:
#     # Plot u: horizontal component of velocity
#     plt.pcolor(turb_lcs.velocity_fields[time][2])
#     plt.savefig('plots/u_interp_t{time}.png'.format(time=time))
#     plt.show()
#
#     # Plot v: vertical component of velocity
#     plt.pcolor(turb_lcs.velocity_fields[time][3])
#     plt.savefig('plots/v_interp_t{time}.png'.format(time=time))
#     plt.show()


# 2. COMPUTE FTLE

# FTLE integration parameters
ftle_dt = -dt_data  # negative for backward-time FTLE
integration_time = 0.6  # integration time in seconds

# Adjust start and end times for calculating FTLE so that enough data is available to integrate
if min_frame is None:
    min_frame = 0
if max_frame is None:
    max_frame = len(time_array_data)

if ftle_dt < 0:
    # If calculating backward FTLE, need to start at least one integration time after beginning of data
    start_frame = min_frame - integration_time / dt_data
    start_time = min_frame * dt_data - integration_time
    end_frame = max_frame - 1
    end_time = (max_frame - 1) * dt_data
if ftle_dt > 0:
    # If calculating forward FTLE, need to end at least one integration time before end of data
    start_frame = min_frame
    start_time = min_frame * dt_data
    end_frame = max_frame - integration_time / dt_data
    end_time = max_frame * dt_data - integration_time

# Evolution time - can manually input different start/end frames here instead of calculating based on data
# tau_list = np.linspace(start_time, end_time, int(abs(((end_time - start_time)/ftle_dt)))+1)

# For testing, just use a snapshot:
#tau_list = [(end_time-start_time/2)]
start_time = min_frame * dt_data
# tau_list = [start_time]
tau_list = [3.78]
# tau_list = np.linspace(start_time, 4, int((4-start_time)/0.02)+1)

# Compute flow map over integration time (and time calculations) - use w_trajs version for FSLE
start_timer = time.time()
# turb_lcs.compute_flow_map_w_trajs(integration_time, tau_list, dt=ftle_dt, method='IE')
turb_lcs.compute_flow_map(integration_time, tau_list, dt=ftle_dt, method='IE')
end_timer = time.time() - start_timer
print('time to compute flow map is: ' + str(end_timer))

# Compute FTLE using central differencing for strain tensor
start_timer = time.time()
turb_lcs.compute_ftle()
# OR compute FSLE
# turb_lcs.compute_fsle(r=5)
print('time to compute FTLE is: ' + str(time.time()-start_timer))

# Create movie of FTLE field, passing in xlim and ylim for plotting. Saves ftle.mp4 in \plots\ folder.
# turb_lcs.ftle_movie((min(xvec_ftle), max(xvec_ftle)), (min(yvec_ftle), max(yvec_ftle)))

# Save LCS & FTLE dictionaries using pickle
# start_timer = time.time()
# fname_lcs = (f'data/LCS_data/alcs_backwardT{integration_time}_spacing{particle_spacing}_0.6to10s_'
#          f'x{min_x_index}to{max_x_index}_y{min_y_index}to{max_y_index}.pickle')
# with open(fname_lcs, 'wb') as handle:
#     pickle.dump(turb_lcs.lcs_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)

# fname_ftle = (f'data/LCS_data/ftle_backwardT{integration_time}_spacing{particle_spacing}_0.6to10s_'
#          f'x{min_x_index}to{max_x_index}_y{min_y_index}to{max_y_index}')
# fname_ftle = 'data/LCS_data/ftletest.pickle'
# with open(fname_ftle, 'wb') as handle:
#     pickle.dump(turb_lcs.ftle, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f'time to save dictionaries: {time.time() - start_timer}')


# For testing, plot snapshot figures:
turb_lcs.ftle_snapshot(tau_list[0], name=f'ftle_odor0.5_backwardT{integration_time}_spacing{particle_spacing}_t{tau_list[0]}'
                                         f'_x{min_x_index}to{max_x_index}_y{min_y_index}to{max_y_index}',
                       odor=None, lcs=False, type='FTLE')
#turb_lcs.plot_lyptime(tau_list[0], name='t0_r5')
#turb_lcs.ftle_snapshot(tau_list[1], name='t2_5', odor=[odor_a, odor_b])
#turb_lcs.ftle_snapshot(tau_list[2], name='t5', odor=[odor_a, odor_b])

