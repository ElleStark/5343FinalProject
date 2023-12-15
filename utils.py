
# Particle binning function in 2D
# copied from https://stackoverflow.com/questions/61325586/fast-way-to-bin-a-2d-array-in-python
def bin2d(orig_array, bin_width):
    m_bins = orig_array.shape[0]//bin_width
    n_bins = orig_array.shape[1]//bin_width
    return orig_array.reshape(m_bins, bin_width, n_bins, bin_width).sum(3).sum(1)


# Below code for Runge Kutta calculations from Dr.Steve Brunton:
# https://www.youtube.com/watch?v=LRF4dGP4xeo

def rk4singlestep(fun, dt, t0, y0):
    """
    Single step of 4th-order Runge-Kutta integration. Use instead of scipy.integrate.solve_ivp to allow for
    vectorized computation of bundle of initial conditions.
    :param fun:
    :param dt:
    :param t0:
    :param y0:
    :return:
    """
    f1 = fun(t0, y0)
    f2 = fun(t0 + dt / 2, y0 + (dt / 2) * f1)
    f3 = fun(t0 + dt / 2, y0 + (dt / 2) * f2)
    f4 = fun(t0 + dt, y0 + dt * f3)
    yout = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    return yout
