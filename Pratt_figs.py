"""
Replicate figures from Pratt et al., 2015, available at https://doi.org/10.1063/1.4914467
Elle Stark Fall 2023

Project organization inspired by C++ implementation for LCS analysis found here: https://github.com/stevenliuyi/lcs
Code inspired by Python FTLE calculations used by: 1) Liu et al., 2018 (https://doi.org/10.1002/2017JC013390):
https://github.com/stevenliuyi/ocean-ftle and 2) https://github.com/jollybao/LCS
"""
import matplotlib.pyplot as plt
import numpy as np
import flowfield

# Constants for double gyre:
a = 0.1  # velocity magnitude A aka U in Pratt et al., 2015
eps = 0.25
T_0 = 10
t = np.linspace(0, 3.5*T_0, 71, endpoint=True)  # use 3.5T_0 as max t to match Pratt et al., 2015

# Create double gyre object and calculate velocity fields
n = 50  # number of grid steps in the x direction, fewer when showing velocity arrows
DoubleGyre = flowfield.DoubleGyre(a, eps, T_0, n)
DoubleGyre.compute_vfields(t)
# TEST: Plot velocity field at a few times
plt.quiver(*DoubleGyre.velocity_fields[0.75*T_0])
plt.show()

# Find flow map using Runge-Kutta 4th order method




# import numpy as np
# import matplotlib.pyplot as plt
#
# x,y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))
#
# u = -y/np.sqrt(x**2 + y**2)
# v = x/np.sqrt(x**2 + y**2)
#
# plt.quiver(x,y,u,v)
# plt.show()

# test Yi Liu's plotting script:
# import numpy as np
# import matplotlib as plt
#
# ftle = np.genfromtxt('double_gyre_ftle_neg.txt', skip_header=3).reshape((1000, 500))
# x = np.linspace(0, 2, num=1000)
# y = np.linspace(0, 1, num=500)
# X, Y = np.meshgrid(x, y, indexing='ij')
# plt.contourf(X, Y, ftle, 100, cmap=plt.cm.Reds)
# plt.show()

# End test




