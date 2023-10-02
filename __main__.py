"""
Replicate figures from Pratt et al., 2015, available at https://doi.org/10.1063/1.4914467
Elle Stark Fall 2023

References for methods:


Project organization inspired by C++ implementation for LCS analysis found here: https://github.com/stevenliuyi/lcs
Code inspired by Python FTLE calculations used by: 1) Liu et al., 2018 (https://doi.org/10.1002/2017JC013390):
https://github.com/stevenliuyi/ocean-ftle and 2) https://github.com/jollybao/LCS
"""
import matplotlib.pyplot as plt
import numpy as np
import flowfield
import time

# Constants for double gyre:
a = 0.1  # velocity magnitude A aka U in Pratt et al., 2015
eps = 0.25
T_0 = 10

# Create double gyre object and calculate velocity fields
n = 400  # number of grid steps in the x direction, use fewer when plotting velocity arrows
DoubleGyre = flowfield.DoubleGyre(a, eps, T_0, n)


### Check: Plot velocity field at a few times
#t = np.linspace(0, 3.5*T_0, 71, endpoint=True)  # use 3.5T_0 as max t to match Pratt et al., 2015
#DoubleGyre.compute_vfields(t)
#plot_times = [0, 0.25, 0.75]
#for time in plot_times:
#    plt.quiver(*DoubleGyre.velocity_fields[time*T_0])
#    plt.show()

# Find flow map using Runge-Kutta 4th order method to integrate backwards from t = t0 to t = t0-T
# Pratt et al. used integration time of 2-2.5 turnover times
T = -2*T_0  # integration time
tau_list = np.linspace(0, T_0, 100)  # evolution times
start_time = time.time()
DoubleGyre.compute_flow_map(T, tau_list)  # note that compute_flow_map also assigns dictionary of self.trajectories
print('time to compute trajectories is: ' + str(time.time()-start_time))

### Check: plot trajectories if small enough to be stored (use method 'compute_flow_map_w_trajs')
#DoubleGyre.plot_trajectories([0, 2], [0, 1])

# Compute FTLE: approximate Right Cauchy-Green strain tensor using central differencing, then find max eigenvalues
# and plug into ftle equation
start_time = time.time()
DoubleGyre.compute_ftle()
print('time to compute FTLE is: ' + str(time.time()-start_time))

# Create movie of FTLE field, passing in xlim and ylim for plotting. Saves ftle.mp4 in \plots\ folder.
DoubleGyre.ftle_movie((0, 2), (0, 1))

# Plot FTLE field at a single point in time. Saves ftle_snap.png in \plots\ folder.
# DoubleGyre.ftle_snapshot(tau_list[-1])

# Save individual FTLE field data if desired
# np.savetxt('data/doublegyre_negftle_t2.5T_T2T0_pratt.txt', DoubleGyre.ftle[tau_list[1]])

