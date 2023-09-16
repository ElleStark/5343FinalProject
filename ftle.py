"""
Module for computing FTLE (finite time Lyapunov exponent) based on a given flow map.
"""


class FTLE:
    """Class for representing FTLE scalar fields from a flow map"""
    def __init__(self, vfield):
        self.vfield = vfield