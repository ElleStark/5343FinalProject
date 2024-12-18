"""
Module for computing FTLE (finite time Lyapunov exponent) based on a given flow map.
[REPURPOSED AS A TESTING SCRIPT]
"""

import numpy as np
import matplotlib.pyplot as plt

ftle_array = np.load('data/LCS_data/FTLE_T0_6_fine_0s_vel_extend_test_200x200.npy')
ftle_array = np.squeeze(ftle_array)

fig, ax = plt.subplots()      
# Get desired FTLE snapshot data
plt.contourf(ftle_array, 100, cmap=plt.cm.Greys, vmin=0, vmax=8)
#plt.title('Odor (red) overlaying FTLE (gray lines)')
plt.colorbar()

ax.set_aspect('equal', adjustable='box')

#plt.savefig('plots/{type}_snap_{name}.png'.format(type=type, name=name), dpi=300)

plt.show()

