"""
Elle Stark October 2023
Expand FTLE computation code to handle discrete velocity fields.
"""

import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import flowfield

# 1. OBTAIN DATA

# Desired time limits for reading in only a subset of the data.
# SET TO None IF ALL TIMES DESIRED!
min_frame = 0
max_frame = 250  # 250 frames is 5 seconds of data for 50Hz resolution

start = time.time()  # track amount of time to read in the data

# Read in all needed data from hdf5 files. For more info on hdf5 format, see https://www.hdfgroup.org/.
# Use 'with' context manager to make sure h5 file is closed properly after using.
# Use [] to store data in memory, and use .item() method to convert single values from array to float.
with h5py.File('D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5', 'r') as f:

    # Metadata: spatiotemporal resolution and domain size
    dt_freq = f.get('Model Metadata/timeResolution')[0].item()
    dt_data = 1 / dt_freq  # convert from Hz to seconds
    time_array_data = f.get('Model Metadata/timeArray')[:]
    spatial_res = f.get('Model Metadata/spatialResolution')[0].item()
    domain_size = f.get('Model Metadata/domainSize')
    domain_width = domain_size[0].item()  # [m] cross-stream distance
    domain_length = domain_size[1].item()  # [m] stream-wise distance

    # Determine total number of frames in the data and assign limits for reading data if not specified by user.
    if min_frame is None:
        min_frame = 0
    if max_frame is None:
        max_frame = len(time_array_data)

    # Numeric grids
    xmesh_uv = f.get('Model Metadata/xGrid')[:].T
    ymesh_uv = f.get('Model Metadata/yGrid')[:].T

    # Velocities: for faster reading, can read in subset of u and v data here
    u_data = f.get('Flow Data/u')[min_frame:max_frame].T  # dimensions (time, columns, rows) = (3001, 1001, 846)
    v_data = f.get('Flow Data/v')[min_frame:max_frame].T  # dimensions (time, columns, rows) = (3001, 1001, 846)

    # Odor data - select from odor pairs (cxa, cxb), where x is integers [1,...,8]
    # higher numbers indicate increasing distance from each other. Source locations are symmetric about y=0.
    odor_a = f.get('Odor Data/c1a')[min_frame:max_frame].T  # location of c1a = [0,0.00375]
    odor_b = f.get('Odor Data/c1b')[min_frame:max_frame].T  # location of c1b = [0, -0.00375]

# Track and display how long it took to read in the data
total_time = time.time()-start
print('time to read in data: ' + str(total_time))

# QC Plot: udata and vdata for comparing to interpolated figure
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
# particle_spacing = spatial_res / 2  # can determine visually if dx is appropriate based on smooth contours for FTLE
# Test with very coarse grid
particle_spacing = spatial_res * 4

# x and y vectors based on velocity mesh limits and particle spacing
xvec_ftle = np.linspace(xmesh_uv[0][0], xmesh_uv[0][-1], int(domain_length / particle_spacing + 1))
yvec_ftle = np.linspace(ymesh_uv[0][0], ymesh_uv[-1][0], int(domain_width / particle_spacing + 1))
xmesh_ftle, ymesh_ftle = np.meshgrid(xvec_ftle, yvec_ftle, indexing='xy')

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
integration_time = -0.06  # integration time in seconds

# Adjust start and end times for calculating FTLE so that enough data is available to integrate
if integration_time < 0:
    # If calculating backward FTLE, need to start at least one integration time after beginning of data
    start_frame = min_frame - integration_time / dt_data
    start_time = min_frame * dt_data - integration_time
    end_frame = max_frame - 1
    end_time = (max_frame - 1) * dt_data
if integration_time > 0:
    # If calculating forward FTLE, need to end at least one integration time before end of data
    start_frame = min_frame
    start_time = min_frame * dt_data
    end_frame = max_frame - integration_time / dt_data
    end_time = max_frame * dt_data - integration_time

# Evolution time - can manually input different start/end frames here instead of calculating based on data
tau_list = np.linspace(start_time, end_time, int(abs(((end_time - start_time)/ftle_dt)))+1)

# Compute flow map over integration time (and time calculations)
start_timer = time.time()
turb_lcs.compute_flow_map(integration_time, tau_list, dt=ftle_dt, method='IE')
end_timer = time.time() - start_timer
print('time to compute flow map is: ' + str(end_timer))

# Compute FTLE using central differencing for strain tensor
start_timer = time.time()
turb_lcs.compute_ftle()
print('time to compute FTLE is: ' + str(time.time()-start_timer))

# Create movie of FTLE field, passing in xlim and ylim for plotting. Saves ftle.mp4 in \plots\ folder.
turb_lcs.ftle_movie((min(xvec_ftle), max(xvec_ftle)), (min(yvec_ftle), max(yvec_ftle)))

