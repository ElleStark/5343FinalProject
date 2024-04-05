"""
Plot the FTLE field at high quality for checking and printing
Elle Stark Mar 2024
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Obtain numpy file with ftle data
ftle_array = np.load('data/LCS_data/FTLE_T0_6_fine_10to14s.npy')

# Obtain datasets from h5 file
time_idx = 194
time_idx_ftle = time_idx - 35
odor = 'c1a'
filename = 'D:/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
with h5py.File(filename, 'r') as f:
    # x and y grids for plotting
    x_grid = f.get(f'Model Metadata/xGrid')[:].T
    y_grid = f.get(f'Model Metadata/yGrid')[:].T
    x_grid_ftle = f.get('Flow Data/LCS/backwardsTime/LCS_mesh/xLCS_mesh_fine')[:].T
    y_grid_ftle = f.get('Flow Data/LCS/backwardsTime/LCS_mesh/yLCS_mesh_fine')[:].T

    # Odor and ftle data
    odor = f.get(f'Odor Data/{odor}')[time_idx, :, :].transpose(1, 0)
    ftle = f.get(f'Flow Data/LCS/backwardsTime/FTLE_60s_T0_6_fine')[time_idx, :, :]

print(f'ftle')

# Flip ftle upside-down spatially to line up with odor field
ftle = np.flip(ftle, axis=0)

# Plot at high resolution
fig, ax = plt.subplots(figsize=(5, 5))
plt.contourf(x_grid_ftle, y_grid_ftle, ftle, 100, cmap=plt.cm.Greys, vmin=0)
plt.colorbar()
plt.pcolormesh(x_grid, y_grid, odor, cmap=plt.cm.Reds, alpha=0.5, vmax=0.5)
ax.set_aspect('equal', adjustable='box')

name = 't13.88s_hi_res_poster'
# plt.savefig(f'plots/ftle_snap_{name}.png', dpi=600)
plt.show()

