"""
Classes for various flow map objects, including:
- Double Gyre
- Discrete flows
Above classes are subclasses of AnalyticalFlow which is subclass of FlowField
"""

import numpy as np
import numpy.linalg as LA
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
#import cv2


class FlowField:
    def __init__(self):
        #super().__init__()
        # Attributes that can be defined by methods
        self.velocity_fields = None
        self.flow_map = None
        self.trajectories = None

    def improvedEuler_singlestep(self, dt, t0, y0):
        """
        Single step of 2nd-order improved Euler integration. vfield must be a function that returns an array of [u, v] values
        :param dt: scalar value of desired time step
        :param t0: start time for integration
        :param y0: starting position of particles
        :return: final position of particles
        """
        # get the slopes at the initial and end points
        f1 = self.vfield(t0, y0)
        f2 = self.vfield(t0 + dt, y0 + dt * f1)
        y_out = y0 + dt / 2 * (f1 + f2)

        return y_out

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
        grid_height = len(self.y[:, 0])
        grid_width = len(self.x[0, :])
        delta_x = self.x[0][1] - self.x[0][0]  # Even spacing, so just take difference at any index
        delta_y = self.y[1][0] - self.y[0][0]

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
                    ftle[j][i] = 1 / (2 * abs(self.integration_time)) * log(sqrt(max_eig))

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

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vLCS = cv2.VideoWriter('Re100_16source_backwardFTLE_T06sec_fine_short_RK4.mp4', fourcc, 20.0, (1000, 800))




        fig, ax = plt.subplots()
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect('equal', adjustable='box')
        # First snapshot
        ax.contourf(x, y, ftle_list[0], 100, cmap=plt.cm.Greys_r)
        #ftle_plot = ax.pcolormesh(x, y, ftle_list[0], cmap=plt.cm.Greys)

        def update(frame):
            for c in ax.collections:
               c.remove()
            ax.contourf(x, y, ftle_list[frame], 100, cmap=plt.cm.Greys)
            #ftle_plot.set_array(ftle_list[frame].ravel())
            return ftle_plot

        ftle_movie = animation.FuncAnimation(fig=fig, func=update, frames=len(ftle_list), interval=200)

        # save video
        f = r"plots/ftle.mp4"
        writervideo = animation.FFMpegWriter(fps=60)
        ftle_movie.save(f, writer=writervideo)

    def ftle_snapshot(self, time, name='1', odor=False):

        # Get desired FTLE snapshot data
        ftle = self.ftle[time]

        # Plot contour map of FTLE
        fig, ax = plt.subplots()
        plt.contourf(self.x, self.y, ftle, 100, cmap=plt.cm.Greys)
        ax.set_aspect('equal', adjustable='box')

        # If odor is True, overlay odor data


        # Save figure
        plt.savefig('plots/ftle_snap_{name}.png'.format(name=name))

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

    def compute_flow_map(self, T, tau_list, dt=None, method='RK4'):
        """
        Uses either Improved Euler or Runge Kutta 4th order method to find flow map from velocity field
        :param T: integration time for particle advection.
        :param tau_list: List of times at which to calculate the flow map. Presumably one FTLE field snapshot will be
                calculated for each tau value.
        :param dt: integration timestep (length of time for each step of the advection algorithm).
        :param method: one of 'RK4' (Runge-Kutta 4th Order) or 'IE' (Improved Euler 2nd order). Defaults to RK4.
        :return: assigns self.flow_map, a dictionary of final particle position arrays with one array per tau
        """
        # keep track of integration time for use in FTLE calculations
        self.integration_time = T

        # Set up variables
        if dt is None:
            dt = T / 1000

        if method == 'RK4':
            advect = self.rk4singlestep
        elif method == 'IE':
            advect = self.improvedEuler_singlestep

        L = abs(int(T / dt))  # need to calculate if dt definition is not based on T
        nx = len(self.x[0, :])
        ny = len(self.y[:, 0])
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

            for step in range(L):
                tstep = step * dt + tau
                yout = advect(dt, tstep, yin)
                yin = yout

            # Final position used for creating flow map
            fmap = yout
            fmap = np.squeeze(fmap)
            fmap_dict[tau] = fmap

        self.flow_map = fmap_dict

    def compute_flow_map_w_trajs(self, T, tau_list, dt=None, method='RK4'):
        """
        Uses either Improved Euler or Runge Kutta 4th order method to find flow map from velocity field.
        ALSO stores positions at each step, so entire particle trajectories can be tracked.
        :param T: integration time for particle advection.
        :param tau_list: List of times at which to calculate the flow map. Presumably one FTLE field snapshot will be
                calculated for each tau value.
        :param dt: integration timestep (length of time for each step of the advection algorithm).
        :param method: one of 'RK4' (Runge-Kutta 4th Order) or 'IE' (Improved Euler 2nd order). Defaults to RK4.
        :return: assigns self.flow_map, a dictionary of final particle position arrays with one array per tau
        """
        # keep track of integration time for use in FTLE calculations
        self.integration_time = T

        # Set up variables
        if dt is None:
            dt = T / 1000

        if method == 'RK4':
            advect = self.rk4singlestep
        elif method == 'IE':
            advect = self.improvedEuler_singlestep

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
                yout = advect(dt, tstep, yin)
                yin = yout
                y_single_steps[:, step, :] = yout

            # Trajectories for all time steps
            self.trajectories = y_single_steps

            # Final position used for creating flow map
            fmap = y_single_steps[:, -1, :]
            fmap = np.squeeze(fmap)
            fmap_dict[tau] = fmap

        self.flow_map = fmap_dict

class DoubleGyre(FlowField):

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
        :param y: array of particle locations where y[0] is array of x locations and y[1] is array of y locations
        :param time: scalar value for time
        :return: array of u and v, where u is size x by y ndarray of horizontal velocity magnitudes,
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

class DiscreteFlow(FlowField):

    def __init__(self, xmesh, ymesh, u_data, v_data, xmesh_uv, ymesh_uv, dt_uv):
        super().__init__()

        self.x = xmesh
        self.y = ymesh
        self.u_data = u_data
        self.v_data = v_data
        self.xmesh_uv = xmesh_uv
        self.ymesh_uv = ymesh_uv
        self.dt_uv = dt_uv

    def vfield(self, time, y):
        """
        Calculates velocity field based on interpolation from existing data.
        :param y: array of particle locations where y[0] is array of x locations and y[1] is array of y locations
        :param time: scalar value for time
        :return: array of u and v, where u is size x by y ndarray of horizontal velocity magnitudes,
        and v is size x by y ndarray of vertical velocity magnitudes.
        """
        # Convert from time to frame
        frame = int(time / self.dt_uv)

        # axes must be in ascending order, so need to flip y-axis, which also means flipping u and v upside-down
        ymesh_vec = np.flipud(self.ymesh_uv)[:, 0]
        xmesh_vec = self.xmesh_uv[0, :]

        # Set up interpolation functions
        # can use cubic interpolation for continuity of the between the segments (improve smoothness)
        # set bounds_error=False to allow particles to go outside the domain by extrapolation
        u_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(self.u_data[:, :, frame])),
                                           method='linear', bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((ymesh_vec, xmesh_vec), np.squeeze(np.flipud(self.v_data[:, :, frame])),
                                           method='linear', bounds_error=False, fill_value=None)

        # Interpolate u and v values at desired x (y[0]) and y (y[1]) points
        u = u_interp((y[1], y[0]))
        v = v_interp((y[1], y[0]))

        vfield = np.array([u, v])

        return vfield

