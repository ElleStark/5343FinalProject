"""
Classes for various flow map objects, including:
- Double Gyre

Above classes are subclasses of AnalyticalFlow which is subclass of FlowField
"""

import numpy as np
import numpy.linalg as LA
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FlowField:
    def rk4singlestep(self, dt, t0, y0):
        """
        Single step of 4th-order Runge-Kutta integration. Use instead of scipy.integrate.solve_ivp to allow for
        vectorized computation of bundle of initial conditions. Reference: https://www.youtube.com/watch?v=LRF4dGP4xeo
        Note that self.vfield must be a function that returns an array of [u, v] values
        :param dt: scalar value of desired time step
        :param t0: start time for integration
        :param y0: starting position of particles
        :return: final position of particles
        """
        # RK4 first computes velocity at full steps and partial steps
        f1 = self.vfield(t0, y0)
        f2 = self.vfield(t0 + dt / 2, y0 + (dt / 2) * f1)
        f3 = self.vfield(t0 + dt / 2, y0 + (dt / 2) * f2)
        f4 = self.vfield(t0 + dt, y0 + dt * f3)
        # RK4 then takes a weighted average to move the particle
        y_out = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return y_out

    def compute_ftle(self):
        """
        modified from https://github.com/jollybao/LCS/blob/master/src/FTLE.py
        Must run method 'compute_flow_map' before using this method.
        :return: assigns self.ftle as dictionary of ftle values (one ftle field per time)
        """
        # Find height and width, and deltas of domain
        grid_height = len(self.yvals)
        grid_width = len(self.xvals)
        delta_x = self.xvals[1] - self.xvals[0]  # Even spacing, so just take difference at any index
        delta_y = self.yvals[1] - self.yvals[0]

        # Initialize dictionary for FTLE fields
        ftle_dict = {}

        for (time, fmap) in self.flow_map.items():
            # Initialize arrays for jacobian approximation and ftle
            jacobian = np.empty([2, 2], float)
            ftle = np.zeros([grid_height, grid_width], float)

            # Use flow map to assign x and y final positions
            x_final = fmap[0]
            x_final = x_final.reshape(grid_height, grid_width)
            y_final = fmap[1]
            y_final = y_final.reshape(grid_height, grid_width)

            # Loop through positions and calculate ftle at each point
            # Leave borders equal to zero (central differencing needs adjacent points for calculation)
            for i in range(1, grid_width - 1):
                for j in range(1, grid_height - 1):
                    jacobian[0][0] = (x_final[j, i + 1] - x_final[j, i - 1]) / (2 * delta_x)
                    jacobian[0][1] = (x_final[j + 1, i] - x_final[j - 1, i]) / (2 * delta_y)
                    jacobian[1][0] = (y_final[j, i + 1] - y_final[j, i - 1]) / (2 * delta_x)
                    jacobian[1][1] = (y_final[j + 1, i] - y_final[j - 1, i]) / (2 * delta_y)

                    # Cauchy-Green tensor
                    gc_tensor = np.dot(np.transpose(jacobian), jacobian)
                    # its largest eigenvalue
                    lamda = LA.eigvals(gc_tensor)
                    max_eig = max(lamda)
                    ftle[j][i] = 1 / (2 * self.integration_time) * log(max_eig)

            ftle_dict[time] = ftle

        self.ftle = ftle_dict

    def ftle_movie(self, xlim, ylim):
        """
        Creates animation from dictionary of ftle values.
        Must call compute_flow_map then compute_ftle before using this method.
        :return: saves .mp4 of ftle evolution
        """
        # Get ftle fields as list - should be in order by ascending time
        ftle_list = list(self.ftle.values())
        x = self.x
        y = self.y

        fig, ax = plt.subplots()
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect('equal', adjustable='box')
        # First snapshot
        ax.contourf(x, y, ftle_list[0], 100, cmap=plt.cm.Greys_r)

        def update(frame):
            for c in ax.collections:
                c.remove()
            ax.contourf(x, y, ftle_list[frame], 100, cmap=plt.cm.Greys_r)

        ftle_movie = animation.FuncAnimation(fig=fig, func=update, frames=len(ftle_list), interval=200)

        # save video
        f = r"plots/ftle.mp4"
        writervideo = animation.FFMpegWriter(fps=60)
        ftle_movie.save(f, writer=writervideo)

    def plot_trajectories(self, xlim, ylim):
        """
        Creates movie of particle trajectories as assigned in compute_flowmap method
        Must call compute_flow_map before using this method.
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

    def compute_flow_map(self, T, tau_list):
        """
        Uses Runge Kutta 4th order method to find flow map from velocity field
        :return:
        """
        # keep track of integration time for use in FTLE calculations
        self.integration_time = T
        dt = T / 1000
        L = abs(int(T / dt))  # need to calculate if dt definition is not based on T
        nx = len(self.xvals)
        ny = len(self.yvals)
        fmap_dict = {}

        # Se up Initial Conditions
        x0 = self.x
        y0 = self.y
        yIC = np.zeros((2, nx * ny))
        yIC[0, :] = x0.reshape(nx * ny)
        yIC[1, :] = y0.reshape(nx * ny)

        # Compute Trajectories
        for tau in tau_list:
            yin = yIC
            y_single_steps = np.zeros((2, L, nx * ny))

            for step in range(L):
                tstep = step * dt + tau
                yout = self.rk4singlestep(dt, tstep, yin)
                yin = yout
                #y_single_steps[:, step, :] = yout

            # Trajectories for all time steps
            #self.trajectories = y_single_steps

            # Final position used for creating flow map
            #fmap = y_single_steps[:, -1, :]
            fmap = yout
            fmap = np.squeeze(fmap)
            fmap_dict[tau] = fmap

        self.flow_map = fmap_dict


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
