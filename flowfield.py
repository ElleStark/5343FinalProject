"""
Classes for various flow map objects, including:
- Double Gyre
"""

import numpy as np
from math import *
class DoubleGyre:

    def __init__(self, a, epsilon, T_0, n):
        self.a = a  # velocity magnitude A aka U
        self.epsilon = epsilon
        self.T_0 = T_0
        self.x = np.linspace(0, 2, num=n)
        self.y = np.linspace(0, 1, num=int(n/2))
        self.velocity_fields = None

    def f(self, x, t):
        '''
        function within double gyre velocity field equations
        :param x: np.array of x values
        :param t: scalar time value
        :return: value for f for use in velocity equations
        '''
        f = x*[1+self.epsilon*np.sin(2*pi*t/self.T_0)*(x-2)]
        df = 1 + 2 * self.epsilon * (x - 1) * np.sin(2 * pi * t / self.T_0)
        return f, df

    def compute_vfields(self, t):
        """
        Computes spatial velocity field for desired times
        :param t: list of time values at which velocity will be calculated
        :return: dictionary of velocity fields for all times stored in object
        """
        x, y = np.meshgrid(self.x, self.y, indexing='xy')
        vfields = []

        # Loop through time, assigning velocity field as x, y, u, v for each t
        for time in t:
            f, df = self.f(x, time)
            u = -pi*self.a*np.sin(pi*f)*np.cos(pi*y)
            v = pi*self.a*np.cos(pi*f)*np.sin(pi*y)*df
            u = np.squeeze(u)  # get rid of extra dimension
            v = np.squeeze(v)

            vfield = [x, y, u, v]
            vfields.append(vfield)
        vfield_dict = dict(zip(t, vfields))

        self.velocity_fields = vfield_dict

    def vfield_elements_at_t(self, t):
        vfield = self.velocity_fields[t]



