"""
Classes for various flow map objects, including:
- Double Gyre

Above classes are subclasses of AnalyticalFlow which is subclass of FlowField
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp


class FlowField:
    def rk4singlestep(self, dt, t0, y0):
        """
        Single step of 4th-order Runge-Kutta integration. Use instead of scipy.integrate.solve_ivp to allow for
        vectorized computation of bundle of initial conditions.
        :param fun:
        :param dt:
        :param t0:
        :param y0:
        :return:
        """
        # vfield must return array of [u, v]
        f1 = self.vfield(t0, y0)
        f2 = self.vfield(t0 + dt / 2, y0 + (dt / 2) * f1)
        f3 = self.vfield(t0 + dt / 2, y0 + (dt / 2) * f2)
        f4 = self.vfield(t0 + dt, y0 + dt * f3)
        yout = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return yout

    def plot_trajectories(self, xlim, ylim):
        """
        Creates movie of particle trajectories as assigned in compute_flowmap method
        Must call compute_flowmap before using this method.
        :param xlim: x-axis limits in form [min, max]
        :param ylim: y-axis limits in form [min, max]
        :return: saves .mp4 animation of particle trajectories in /plots folder
        """

        x_trajs = self.trajectories[0]
        y_trajs = self.trajectories[1]
        # set up figure
        fig, ax = plt.subplots()
        # First snapshot
        positions = ax.scatter(x_trajs, y_trajs, s=0.1, c='black')
        # Plotting configuration
        ax.set(xlim=xlim, ylim=ylim, xlabel='x', ylabel='y')

        def init_scatter():
            positions.set_offsets([])
            return(positions,)

        def update(frame):
            data = np.column_stack((x_trajs[frame], y_trajs[frame]))
            positions.set_offsets(data)
            return (positions, )

        # len(DoubleGyre.trajectories[1]) for num frames?
        traj_movie = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=200, blit=True)

        # save video
        f = r"plots/trajectories.mp4"
        writervideo = animation.FFMpegWriter(fps=60)
        traj_movie.save(f, writer=writervideo)

class AnalyticalFlow(FlowField):

    def __init__(self):
        super().__init__()

        # Attributes that can be defined by methods
        self.velocity_fields = None
        self.flow_map = None
        self.trajectories = None

    def compute_vfields(self, t):
        """
        Computes spatial velocity field for list of desired times
        :param t: ndarray of time values at which velocity field will be calculated
        :return: dictionary of velocity fields, one for each time value.
                 Each velocity field is a list of 4 ndarrays: [x, y, u, v].
        """
        vfields = []

        # Loop through time, assigning velocity field [x, y, u, v] for each t
        for time in t:
            vfield = self.vfield(time, [self.x, self.y])
            # need to extract u and v from vfield array
            u = vfield[0]
            v = vfield[1]
            vfield = [self.x, self.y, u, v]
            vfields.append(vfield)
        vfield_dict = dict(zip(t, vfields))

        self.velocity_fields = vfield_dict

    def compute_flow_map(self, T, tau):
        """
        Uses Runge Kutta 4th order method to find flow map from velocity field
        :return:
        """
        dt = T / 1000
        L = abs(int(T / dt))  # need to calculate if dt definition is not based on T
        nx = len(self.xvals)
        ny = len(self.yvals)

        # Se up Initial Conditions
        x0 = self.x
        y0 = self.y
        yIC = np.zeros((2, nx * ny))
        yIC[0, :] = x0.reshape(nx * ny)
        yIC[1, :] = y0.reshape(nx * ny)

        # Compute Trajectories
        # ADD TAU LOOP ONCE CODE IS WORKING IF DESIRED
        yin = yIC
        y_single_steps = np.zeros((2, L, nx * ny))

        for step in range(L):
            tstep = step * dt
            yout = self.rk4singlestep(dt, tstep, yin)
            yin = yout
            y_single_steps[:, step, :] = yout

        # Trajectories for all time steps
        self.trajectories = y_single_steps

        # Final position used for creating flow map
        fmap = y_single_steps[:, -1, :]
        fmap = np.squeeze(fmap)
        self.flow_map = fmap


class DoubleGyre(AnalyticalFlow):

    def __init__(self, a, epsilon, T_0, n):
        super().__init__()

        self.a = a  # velocity magnitude A aka U
        self.epsilon = epsilon
        self.T_0 = T_0
        self.xvals = np.linspace(0, 2, num=n)
        self.yvals = np.linspace(0, 1, num=int(n / 2))
        self.x, self.y = np.meshgrid(self.xvals, self.yvals, indexing='xy')

    def vfield(self, time, y):
        """
        Calculates velocity field based on double gyre analytical equations
        :param x: scalar x value OR ndarray of x values
        :param y: scalar y value OR ndarray of y values
        :param time: scalar value for time
        :return: list of u and v, where u is size x by y ndarray of horizontal velocity magnitudes,
        and v is size x by y ndarray of vertical velocity magnitudes.
        """
        f = y[0] * [1 + self.epsilon * np.sin(2 * pi * time / self.T_0) * (y[0] - 2)]
        df = 1 + 2 * self.epsilon * (y[0] - 1) * np.sin(2 * pi * time / self.T_0)

        u = -pi * self.a * np.sin(pi * f) * np.cos(pi * y[1])
        v = pi * self.a * np.cos(pi * f) * np.sin(pi * y[1]) * df
        u = np.squeeze(u)  # get rid of extra dimension of length 1 if present
        v = np.squeeze(v)

        vfield = np.array([u, v])  # convert to array for vectorization

        return vfield
