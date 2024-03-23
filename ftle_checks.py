"""
Plot the FTLE field at a few times to check the computations performed Mar 2024
Elle Stark Mar 2024
"""

import matplotlib.pyplot as plt
import numpy as np

ftle_array = np.load('data/LCS_data/FTLE_T0_6_fine_35to40s.npy')

idx = 0
plt.contourf(ftle_array[idx], 100, cmap=plt.cm.Greys, vmin=0)
plt.colorbar()
plt.show()

