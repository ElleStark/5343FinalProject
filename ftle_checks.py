"""
Plot the FTLE field at high quality for checking and printing
Elle Stark Mar 2024
"""

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import numpy as np


# Function to create custom color map for plotting
def create_custom_cmap(hex_color):
    cmap_data = [(0, 'white'), (1, hex_color)]
    return LinearSegmentedColormap.from_list('custom_cmap', cmap_data)


# Obtain numpy file with ftle data
ftle_array = np.load('data/LCS_data/FTLE_T0_6_fine_10to14s.npy')

# Obtain datasets from h5 file
time_idx = 190
time_idx_ftle = time_idx - 30
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
    ftle = f.get(f'Flow Data/LCS/backwardsTime/FTLE_60s_T0_6_fine')[time_idx_ftle, :, :]

# Flip ftle upside-down spatially to line up with odor field
ftle = np.flip(ftle, axis=0)

# Create custom colormaps
# o_color = '#7d0a00'  # Darker for transparency version
o_color = '#B85B51'
odor_cmap = create_custom_cmap(o_color)
f_color = '#07090F'
ftle_cmap = create_custom_cmap(f_color)

# Plot at high resolution
fig, ax = plt.subplots(figsize=(10, 10))
# plt.contourf(x_grid_ftle, y_grid_ftle, ftle, 100, cmap=ftle_cmap, vmin=0)
# plt.colorbar(orientation='vertical')
plt.pcolormesh(x_grid, y_grid, odor, cmap=odor_cmap, alpha=1, norm=matplotlib.colors.LogNorm())
plt.colorbar(orientation='vertical')
ax.set_aspect('equal', adjustable='box')
plt.ylim(-0.211, 0.211)
name = 't3.8s_odorplume_poster2'
# plt.savefig(f'plots/ftle_snap_{name}.png', dpi=600)
plt.show()

