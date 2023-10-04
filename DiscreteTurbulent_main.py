"""
Elle Stark October 2023
Expand FTLE computation code to handle discrete velocity fields.
"""

import h5py
import time

import numpy as np

# Desired time limits for reading in only a subset of the data.
# SET TO None IF ALL TIMES DESIRED!
min_frame = 0
max_frame = 250  # 250 frames is 5 seconds of data for 50Hz resolution

start = time.time()  # track amount of time to read in the data

# Read in all needed data from hdf5 files. For more info on hdf5 format, see https://www.hdfgroup.org/.
# Use 'with' context manager to make sure h5 file is closed properly after using
with h5py.File('D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5', 'r') as f:

    # Metadata: spatiotemporal resolution and domain size
    dt_freq = f.get('Model Metadata/timeResolution')[0]
    dt = 1 / dt_freq  # convert from Hz to seconds
    time_array_data = f.get('Model Metadata/timeArray')[:]
    spatial_res = f.get('Model Metadata/spatialResolution')[0]
    domain_size = f.get('Model Metadata/domainSize')
    domain_width = domain_size[0]  # [m] cross-stream distance
    domain_length = domain_size[1]  # [m] stream-wise distance

    # Determine total number of frames in the data and assign limits for reading data if not specified by user.
    if min_frame is None:
        min_frame = 0
    if max_frame is None:
        max_frame = len(time_array_data)

    # Numeric grids
    xmesh_uv = f.get('Model Metadata/xGrid')[:]
    ymesh_uv = f.get('Model Metadata/yGrid')[:]

    # Velocities: for faster reading, can read in subset of u and v data here
    u = f.get('Flow Data/u')[min_frame:max_frame]  # dimensions (time, columns, rows) = (3001, 1001, 846)
    v = f.get('Flow Data/v')[min_frame:max_frame]  # dimensions (time, columns, rows) = (3001, 1001, 846)

    # Odor data - select from odor pairs (cxa, cxb), where x is integers [1,...,8]
    # higher numbers indicate increasing distance from each other. Source locations are symmetric about y=0.
    odor_a = f.get('Odor Data/c1a')[min_frame:max_frame]  # location of c1a = [0,0.00375]
    odor_b = f.get('Odor Data/c1b')[min_frame:max_frame]  # location of c1b = [0, -0.00375]

# Track and display how long it took to read in the data
total_time = time.time()-start
print('time to read in data: ' + str(total_time))

# FTLE integration parameters
ftle_dt = dt
integration_time = 0.6  # integration time in seconds

# Create grid of particles with desired spacing
particle_spacing = spatial_res / 2  # can determine visually if dx is appropriate based on smooth contours for FTLE field
# x and y vectors based on velocity mesh limits and particle spacing
xvec_ftle = np.linspace(xmesh_uv[0][0], xmesh_uv[-1][0], int(domain_length / particle_spacing + 1))
yvec_ftle = np.linspace(ymesh_uv[0][0], ymesh_uv[0][-1], int(domain_width / particle_spacing + 1))

xmesh_ftle, ymesh_ftle = np.meshgrid(xvec_ftle, yvec_ftle, indexing='xy')



