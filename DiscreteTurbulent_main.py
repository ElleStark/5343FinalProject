"""
Elle Stark October 2023
Expand FTLE computation code to handle discrete velocity fields.
"""

import h5py


# Desired domain for reading in only a subset of the data
# limits = xxxxx

# Read in all needed data from hdf5 files. For more info on hdf5 format, see https://www.hdfgroup.org/.
# Use 'with' context manager to make sure h5 file is closed properly after using
with h5py.File('D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5', 'r') as f:
    # Numeric grids
    xmesh_data = f.get('Model Metadata/xGrid')
    ymesh_data = f.get('Model Metadata/yGrid')

    # Velocities: for faster reading, can read in subset of u and v data here
    u_data = f.get('Flow Data/u')  # dimensions (time, columns, rows) = (3001, 1001, 846)
    v_data = f.get('Flow Data/v')  # dimensions (time, columns, rows) = (3001, 1001, 846)

    # Odor data
    odor_a_data = f.get('Odor Data/c1a')
    odor_b_data = f.get('Odor Data/c1b')

    # Metadata: spatiotemporal resolution and domain size
    dt_freq = f.get('Model Metadata/timeResolution')
    time_array_data = f.get('Model Metadata/timeArray')
    spatial_res = f.get('Model Metadata/spatialResolution')
    domain_size = f.get('Model Metadata/domainSize')

    # Copy data from H5 file into memory

    xmesh = xmesh_data
    ymesh = ymesh_data

    u = u_data[:]
    v = v_data[:]

    odor_a = odor_a_data
    odor_b = odor_b_data

    dt = 1 / dt_freq
    time_array = time_array_data
    dx = spatial_res
    domain_width = domain_size[0]  # [m] cross-stream distance
    domain_length = domain_size[1]  # [m] stream-wise distance


print(dx)

