"""
Replicate figures from Pratt et al., 2015, available at https://doi.org/10.1063/1.4914467
Elle Stark Fall 2023

Project organization inspired by C++ implementation for LCS analysis found here: https://github.com/stevenliuyi/lcs
Code inspired by Python FTLE calculations used by: 1) Liu et al., 2018 (https://doi.org/10.1002/2017JC013390):
https://github.com/stevenliuyi/ocean-ftle and 2) https://github.com/jollybao/LCS
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import flowfield

# Constants for double gyre:
a = 0.1  # velocity magnitude A aka U in Pratt et al., 2015
eps = 0.25
T_0 = 10
t = np.linspace(0, 3.5*T_0, 71, endpoint=True)  # use 3.5T_0 as max t to match Pratt et al., 2015

# Create double gyre object and calculate velocity fields
n = 40  # number of grid steps in the x direction, fewer when showing velocity arrows
DoubleGyre = flowfield.DoubleGyre(a, eps, T_0, n)
DoubleGyre.compute_vfields(t)
# Check: Plot velocity field at a few times
plot_times = [0, 0.25, 0.75]
for time in plot_times:
    plt.quiver(*DoubleGyre.velocity_fields[time*T_0])
    plt.show()

# Find flow map using Runge-Kutta 4th order method to integrate backwards from t = t0
# Pratt et al. used integration time of 2-2.5 turnover times
T = -2*T_0  # integration time
tau = [0, 2.5*T_0, 3*T_0, 3.5*T_0]  # evolution time 0 to 3.5 T_0
DoubleGyre.compute_flow_map(T, tau)
#
# ### Check: plot trajectories
# x_trajs = DoubleGyre.trajectories[0]
# y_trajs = DoubleGyre.trajectories[1]
# # set up figure
# fig, ax = plt.subplots()
# # First snapshot
# positions = ax.scatter(x_trajs, y_trajs, s=0.1, c='black')
# # Plotting configuration
# ax.set(xlim=[0, 2], ylim=[0, 1], xlabel='x', ylabel='y')
#
# def init_scatter():
#     positions.set_offsets([])
#     return(positions,)
#
# def update(frame):
#     data = np.column_stack((x_trajs[frame], y_trajs[frame]))
#     positions.set_offsets(data)
#     return (positions, )
#
# # len(DoubleGyre.trajectories[1]) for num frames?
# traj_movie = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=200, blit=True)
# plt.show()
#
# # save and show video
# f = r"plots/trajectories_200.mp4"
# writervideo = animation.FFMpegWriter(fps=60)
# traj_movie.save(f, writer=writervideo)







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




