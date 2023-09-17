"""
Classes for various flow map objects, including:
- Double Gyre
"""

import numpy as np
from math import *
import scipy as sp
import utils

# class flowfield:
#     def compute_flow_map(self):
#         """
#         Uses Runge Kutta 4th order method to find flow map from velocity field
#         :return:
#         """


class DoubleGyre():

    def __init__(self, a, epsilon, T_0, n):
        #super().__init__()

        self.a = a  # velocity magnitude A aka U
        self.epsilon = epsilon
        self.T_0 = T_0
        self.xvals = np.linspace(0, 2, num=n)
        self.yvals = np.linspace(0, 1, num=int(n/2))
        self.x, self.y = np.meshgrid(self.xvals, self.yvals, indexing='xy')
        self.velocity_fields = None
        self.flow_map = None
        self.trajectories = None

    def f(self, x, t):
        '''
        function within double gyre velocity field equations
        :param x: scalar x value OR ndarray of x values
        :param t: scalar time value
        :return: value for f for use in velocity equations
        '''
        f = x*[1+self.epsilon*np.sin(2*pi*t/self.T_0)*(x-2)]
        df = 1 + 2 * self.epsilon * (x - 1) * np.sin(2 * pi * t / self.T_0)
        return f, df

    def vfield(self, x, y, time):
        """
        Calculates velocity field based on double gyre analytical equations
        :param x: scalar x value OR ndarray of x values
        :param y: scalar y value OR ndarray of y values
        :param time: scalar value for time
        :return:
        """
        f, df = self.f(x, time)
        u = -pi * self.a * np.sin(pi * f) * np.cos(pi * y)
        v = pi * self.a * np.cos(pi * f) * np.sin(pi * y) * df
        u = np.squeeze(u)  # get rid of extra dimension of length 1 if present
        v = np.squeeze(v)

        vfield = [u, v]

        return vfield

    def compute_vfields(self, t):
        """
        Computes spatial velocity field for list of desired times
        :param t: ndarray of time values at which velocity field will be calculated
        :return: dictionary of velocity fields, one for each time value
        """
        vfields = []
        # Loop through time, assigning velocity field [x, y, u, v] for each t
        for time in t:
            vfield = self.vfield(self.x, self.y, time)
            vfield = [self.x, self.y] + vfield
            vfields.append(vfield)
        vfield_dict = dict(zip(t, vfields))

        self.velocity_fields = vfield_dict

    def vfield_mod(self, t, y):
        """
        Rewriting velocity field equations to take y0 instead of x, y
        :param y: y[0] is x and y[1] is y
        :param t: scalar value for time
        :return:
        """
        f, df = self.f(y[0], t)

        u = -pi * self.a * np.sin(pi * f) * np.cos(pi * y[1])
        v = pi * self.a * np.cos(pi * f) * np.sin(pi * y[1]) * df
        u = np.squeeze(u)  # get rid of extra dimension of length 1 if present
        v = np.squeeze(v)

        vfield = np.array([u, v])

        return vfield

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
        f1 = self.vfield_mod(t0, y0)
        f2 = self.vfield_mod(t0 + dt / 2, y0 + (dt / 2) * f1)
        f3 = self.vfield_mod(t0 + dt / 2, y0 + (dt / 2) * f2)
        f4 = self.vfield_mod(t0 + dt, y0 + dt * f3)
        yout = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        return yout

    def compute_flow_map(self, T, tau):
        """
        Uses Runge Kutta 4th order method to find flow map from velocity field
        :return:
        """
        dt = T/1000  # for T=10, dt = 0.01
        L = int(T / dt) # need to calculate if dt definition is not based on T
        nx = len(self.xvals)
        ny = len(self.yvals)

        # Se up Initial Conditions
        x0 = self.x
        y0 = self.y
        yIC = np.zeros((2, nx * ny))
        yIC[0, :] = x0.reshape(nx * ny)
        yIC[0, :] = y0.reshape(nx * ny)

        # Compute Trajectories
        # ADD TAU LOOP ONCE CODE IS WORKING!
        yin = yIC
        y_single_steps = np.zeros((2, L, nx * ny))

        for step in range(L):
            tstep = step * dt
            yout = self.rk4singlestep(dt, tstep, yin)
            yin = yout
            y_single_steps[:, step, :] = yout

        self.trajectories = y_single_steps

        fmap = y_single_steps[:, -1, :]
        #fmap = fmap.reshape

        self.flow_map = fmap



