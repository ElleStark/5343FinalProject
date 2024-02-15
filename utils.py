import numpy as np
# Import Rectangular bivariate spline from scipy
from scipy.interpolate import RectBivariateSpline as RBS


def eigen(A):
    """The function computes the eigenvalues and eigenvectors of a two-dimensional symmetric matrix.
    from TBarrier repository by Encinas Bartos, Kaszas, Haller 2023: https://github.com/EncinasBartos/TBarrier
    Parameters:
        A: array(2,2), input matrix


    Returns:
        lambda_min: float, minimal eigenvalue
        lambda_max: float, maximal eigenvalue
        v_min: array(2,), minimal eigenvector
        v_max: array(2,), maximal eigenvector
    """
    A11 = A[0, 0]  # float
    A12 = A[0, 1]  # float
    A22 = A[1, 1]  # float

    discriminant = (A11 + A22) ** 2 / 4 - (A11 * A22 - A12 ** 2)  # float

    if discriminant < 0 or np.isnan(discriminant):
        return np.nan, np.nan, np.zeros((1, 2)) * np.nan, np.zeros((1, 2)) * np.nan

    lambda_max = (A11 + A22) / 2 + np.sqrt(discriminant)  # float
    lambda_min = (A11 + A22) / 2 - np.sqrt(discriminant)  # float

    v_max = np.array([-A12, A11 - lambda_max])  # array (2,)
    v_max = v_max / np.sqrt(v_max[0] ** 2 + v_max[1] ** 2)  # array (2,)

    v_min = np.array([-v_max[1], v_max[0]])  # array (2,)

    return lambda_min, lambda_max, v_min, v_max


def _loc_max(min_distance, X, Y, Field, loc_threshold=0):
    '''
    Wrapper for _find_reduced_set_of_local_maxima().
    Find a set of local maxima in a scalar field which are separated by a set distance and are above a threshold value.
    Returns the peaks sorted according to their field values.

    Parameters:
        max_distance: float, discard maxima which are closer than max_distance to another maximum
        X: array (Nx, Ny), X-meshgrid
        Y: array (Nx, Ny), Y-meshgrid
        Field: array(Nx, Ny), Scalar field to analyze
        loc_threshold: float, consider only maxima above this threshold

    Returns:
        peak_idx_x: X position in the grid of the maxima
        peak_idx_y: Y position in the grid of the maxima
        peak_x: X position of the maxima (peak_x = X[peak_idx_x, peak_idx_y])
        peak_y: Y position of the maxima (peak_y = Y[peak_idx_x, peak_idx_y])
        peak_field: value of the maxima
    '''

    # find local maxima
    peak_x, peak_y, peak_field, peak_idx_x, peak_idx_y = _find_reduced_set_of_local_maxima(min_distance, X, Y, Field,
                                                                                           loc_threshold)

    # sort local maxima in descending order
    arg_index = np.argsort(peak_field)

    # x,y-position of local maximum
    peak_x = np.flip([peak_x[i] for i in arg_index])
    peak_y = np.flip([peak_y[i] for i in arg_index])

    # i,j-index of local maximum
    peak_idx_x = np.flip([peak_idx_x[i] for i in arg_index])
    peak_idx_y = np.flip([peak_idx_y[i] for i in arg_index])

    # value of local maximum
    peak_field = np.flip([peak_field[i] for i in arg_index])

    return peak_idx_x, peak_idx_y, peak_x, peak_y, peak_field


def _find_all_local_maxima(X, Y, Field):
    '''
    Find all local maxima in a scalar field which are separated by a set distance and are above a threshold value.

    Parameters:
        X: array (Nx, Ny), X-meshgrid
        Y: array (Nx, Ny), Y-meshgrid
        Field: array(Nx, Ny), Scalar field to analyze

    Returns:
        idx_x: X position in the grid of the maxima
        idx_y: Y position in the grid of the maxima
        loc_max_x: X position of the maxima (loc_max_x = X[idx_x, idx_y])
        loc_max_y: Y position of the maxima (loc_max_y = Y[idx_x, idx_y])
        loc_max_field: value of the maxima
    '''
    loc_max_x, loc_max_y, loc_max_field = [], [], []
    idx_x, idx_y = [], []

    for i in range(2, X.shape[0] - 2):

        for j in range(2, Y.shape[1] - 2):

            if np.isfinite(Field[i, j]) and Field[i, j] > Field[i + 1, j] and Field[i, j] > Field[i - 1, j] and Field[
                i, j] > Field[i, j + 1] and Field[i, j] > Field[i, j - 1]:
                loc_max_x.append(X[i, j])
                loc_max_y.append(Y[i, j])
                loc_max_field.append(Field[i, j])
                idx_x.append(j)
                idx_y.append(i)

    return idx_x, idx_y, loc_max_x, loc_max_y, loc_max_field


def _find_reduced_set_of_local_maxima(min_distance, X, Y, Field, loc_threshold):
    """
    Find all local maxima in a scalar field which are separated by a set distance and are above a threshold value.

    Parameters:
        X: array (Nx, Ny), X-meshgrid
        Y: array (Nx, Ny), Y-meshgrid
        Field: array(Nx, Ny), Scalar field to analyze

    Returns:
        idx_x: X position in the grid of the maxima
        idx_y: Y position in the grid of the maxima
        loc_max_x: X position of the maxima (loc_max_x = X[idx_x, idx_y])
        loc_max_y: Y position of the maxima (loc_max_y = Y[idx_x, idx_y])
        loc_max_field: value of the maxima
    """
    idx_x, idx_y, loc_max_x, loc_max_y, loc_max_field = _find_all_local_maxima(X, Y, Field)

    n_loc_max = len(loc_max_x)

    peak_x, peak_y, peak_field = [], [], []
    peak_idx_x, peak_idx_y = [], []

    for i in range(n_loc_max):

        bool_loc_max = True

        for j in range(n_loc_max):

            if i != j and loc_max_field[i] < loc_max_field[j] and sqrt(
                    (loc_max_x[i] - loc_max_x[j]) ** 2 + (loc_max_y[i] - loc_max_y[j]) ** 2) <= min_distance:
                bool_loc_max = False

        if bool_loc_max and loc_max_field[i] > loc_threshold:
            peak_x.append(loc_max_x[i])
            peak_y.append(loc_max_y[i])
            peak_field.append(loc_max_field[i])
            peak_idx_x.append(idx_x[i])
            peak_idx_y.append(idx_y[i])

    return peak_x, peak_y, peak_field, peak_idx_x, peak_idx_y


def _scaling_vectorfield_incompressible(X, Y, x, x_prime, vector_field, Interp_eig):
    '''
    Scaling of vectorfield turns tensorlines singularities into fixed points for incompressible vector fields.

    Parameters:
        X:               array (Ny, Nx), X-meshgrid
        Y:               array (Ny, Nx), Y-meshgrid
        x:               array (2, Npoints), position (#Npoints = Number of initial conditions)
        x_prime:         array (2, Npoints), eigenvector at 'x'
        vector_field:    array (Ny, Nx, 2), eigenvector field over domain domain.
        interp_eig:      Interpolant-object for eigenvalue field

    Returns:
        rescaled_vector: array (2, Npoints), rescaled version of eigenvector.
                         If the point is outside of the defined domain, then 'None' is returned
    '''

    vx, vy = _orient_vectorfield(X, Y, x, vector_field)

    if vx is not None:

        # compute lambda_2
        lambda_max = Interp_eig(x[1], x[0])[0][0]

        # if lambda_2 == 0 --> stop integration.
        # This happens in regions close to the boundary, where the incompressibility condition is not satisfied anymore.
        if lambda_max == 0:
            return None

        # assuming incompressibility
        lambda_min = 1 / lambda_max

        # transform singularities to fixed points
        alpha = ((lambda_max - lambda_min) / (lambda_max + lambda_min)) ** 2

        # rescalin
        scaling = np.sign(vx * x_prime[0] + vy * x_prime[1]) * alpha

        rescaled_vector = scaling * np.array([vx, vy])  # array

        return rescaled_vector  # array

    else:
        return None


def _orient_vectorfield(X, Y, x, vector_field):
    '''
    Eigenvector-field is globally non-orientable. However, it can be globally re-oriented along trajectories.
    This function reorients the eigenvector field locally and evaluates the locally re-oriented eigenvector field at 'x'
    via linear interpolation.

    Parameters:
        X, Y:         array(Ny, Nx), gridded X,Y domain
        x:            float, current position of tensorline
        vector_field: array (Ny, Nx, 2), gridded eigenvector field

    Returns:
        vx, vy:       float, x/y-components of linearly interpolated eigenvector field at 'x'
    '''

    # Check for orientational discontinuity by introducing appropriate scaling
    idx_x = np.searchsorted(X[0, :], x[0])  # float
    idx_y = np.searchsorted(Y[:, 0], x[1])  # float

    # If not on the boundary of the domain of the data.
    if 0 < idx_x < X.shape[1] and 0 < idx_y < Y.shape[0]:

        # extract meshgrid in proximity of particle location (=local meshgrid)
        X_reduced, Y_reduced = X[idx_y - 1:idx_y + 1, idx_x - 1:idx_x + 1], Y[idx_y - 1:idx_y + 1,
                                                                            idx_x - 1:idx_x + 1]  # array (2,2)

        # extract vector field in proximity of particle location (=local vector field)
        vx_grid = np.array([[vector_field[idx_y - 1, idx_x - 1, 0], vector_field[idx_y, idx_x - 1, 0]],
                            [vector_field[idx_y - 1, idx_x, 0], vector_field[idx_y, idx_x, 0]]])  # array (2,2)
        vy_grid = np.array([[vector_field[idx_y - 1, idx_x - 1, 1], vector_field[idx_y, idx_x - 1, 1]],
                            [vector_field[idx_y - 1, idx_x, 1], vector_field[idx_y, idx_x, 1]]])  # array (2,2)

        # re-orient the local vector field so that all 4 vectors at the grid-points around x point in the same direction
        for i in range(2):
            for j in range(2):
                if vx_grid[0, 0] * vx_grid[i, j] + vy_grid[0, 0] * vy_grid[i, j] < 0:
                    vx_grid[i, j] = -vx_grid[i, j]  # float
                    vy_grid[i, j] = -vy_grid[i, j]  # float

        # Linearly interpolate vector-field.
        vx_Interp = RBS(Y_reduced[:, 0], X_reduced[0, :], vx_grid, kx=1, ky=1)  # RectangularBivariateSpline object
        vy_Interp = RBS(Y_reduced[:, 0], X_reduced[0, :], vy_grid, kx=1, ky=1)  # RectangularBivariateSpline object

        vx = vx_Interp(x[1], x[0])[0][0]
        vy = vy_Interp(x[1], x[0])[0][0]

        return vx, vy

    # If particle outside of domain of the data --> return None, None
    else:

        return None, None


def check_location(X, Y, defined_domain, x, no_nans_in_domain=False):
    '''This function evaluates the location of the particle at $ \mathbf{x} $.
    It returns the leftsided indices of the meshgrid X, Y where the particle is located.
    Based on the domain where the flow field is defined,
    the location of the particle is categorized either as being
    1. inside the flow domain:"IN";
        This happens at points where the velocity field is locally well defined
        (= The velocities at the four adjacent grid points of the mesh is defined)
    2. outside the flow domain: "OUT";
        This happens at points where the velociy field is not defined at all
        (= The velocities at the four adjacent grid points of the mesh is not defined)
    3. at the boundary: "BOUNDARY";
        This happens at points where the velocity field is only partially defined
        such as at a wall boundary or at the interface between land and sea.

    Parameters:
        X: array(Ny, Nx), X-grid
        Y: array(Ny, Nx), Y-grid
        defined_domain: array(Ny, Nx), points in the grid where the velocity is defined
        x: array(2,), position to querry
        no_nans_in_domain: bool, Guarantee that there aren't any nans in the domain. Default is False

    Returns:
        loc: "IN", "OUT", "BOUNDARY"
        idx_x: indicate the position if there are nans in the domain
        idx_y: indicate the position if there are nans in the domain
    '''
    # Define boundaries
    Xmax = X[0, -1]
    Xmin = X[0, 0]
    Ymin = Y[0, 0]
    Ymax = Y[-1, 0]

    # current position
    xp = x[0]
    yp = x[1]

    # if there are non nans inside the domain, then we can just worry about the boundaries
    if no_nans_in_domain:

        if Xmin < xp < Xmax and Ymin < yp < Ymax:

            loc = "IN"

            return loc, None, None

        else:

            loc = "OUT"

            return loc, None, None

    # if there are nans in the domain (e.g. Land in the ocean), then we need to take that into account
    else:

        # compute left/lower indices of location of the particle with respect to the meshgrid
        idx_x = np.searchsorted(X[0, :], xp)
        idx_y = np.searchsorted(Y[:, 0], yp)

        # check if particle outside rectangular boundaries
        if xp < Xmin or xp > Xmax or yp < Ymin or yp > Ymax or np.isnan(xp) or np.isnan(yp):

            loc = "OUT"

        else:

            # particle at the left boundary of domain
            if idx_x == 0:
                idx_x = 1

            # particle at the lower boundary of domain
            if idx_y == 0:
                idx_y = 1

            Condition_nan = np.sum(defined_domain[idx_y - 1:idx_y + 1, idx_x - 1:idx_x + 1].ravel())

            if Condition_nan == 4:

                loc = "IN"

            else:

                loc = "OUT"

        return loc, idx_x, idx_y


def _RK4_tensorlines_incompressible(X, Y, defined_domain, x, x_prime, step_size, vector_field, interp_eig):
    '''
    Computes tensorlines using RK4-integration scheme.

    Parameters:
        X:              array (Ny, Nx), X-meshgrid
        Y:              array (Ny, Nx), Y-meshgrid
        defined_domain: array (Ny, Nx), grid specifying whether velocity field is defined (=1) or undefined (=0)
        x:              array (2, Npoints), position (#Npoints = Number of initial conditions)
        x_prime:        array (2, Npoints), eigenvector at 'x'
        step_size:      float, step size used for integration. This value is kept constant.
        vector_field:   array (Ny, Nx, 2), eigenvector field over domain domain
        interp_eig:     Interpolant-object for eigenvalue field

    Returns:
        x_update:       array (2, Npoints), updated coordinate of tensorline
        x_prime:        array (2, Npoints), eigenvector at 'x_update' with the same orientation as 'x_prime'
    '''

    # Define starting point.
    x1 = x

    # Check if particle is in domain of data
    loc = check_location(X, Y, defined_domain, x1)[0]

    # If not in domain --> stop integration
    if loc != "IN":
        return None, None

    # Compute x_prime at the beginning of the time-step by re-orienting and rescaling the vector field
    x_prime = _scaling_vectorfield_incompressible(X, Y, x1, x_prime, vector_field, interp_eig)

    # x_prime can be None at the boundaries of the spatial domain of the data --> stop integration
    if x_prime is None:
        return None, None

    # compute derivative
    k1 = step_size * x_prime

    #  position and time at the first midpoint.
    x2 = x1 + .5 * k1

    # Check if particle is in domain of data
    loc = check_location(X, Y, defined_domain, x2)[0]

    # If not in domain --> stop integration
    if loc != "IN":
        return None, None

    # Compute x_prime at the first midpoint.
    x_prime = _scaling_vectorfield_incompressible(X, Y, x2, x_prime, vector_field, interp_eig)

    # x_prime can be None at the boundaries of the spatial domain of the data --> stop integration
    if x_prime is None:
        return None, None

    # compute derivative
    k2 = step_size * x_prime

    # Update position at the second midpoint.
    x3 = x1 + .5 * k2

    # Check if particle is in domain of data
    loc = check_location(X, Y, defined_domain, x3)[0]

    # If not in domain --> stop integration
    if loc != "IN":
        return None, None

    # Compute x_prime at the second midpoint.
    x_prime = _scaling_vectorfield_incompressible(X, Y, x3, x_prime, vector_field, interp_eig)

    # x_prime can be None at the boundaries of the spatial domain of the data --> stop integration
    if x_prime is None:
        return None, None

    # compute derivative
    k3 = step_size * x_prime

    # Update position at the endpoint.
    x4 = x1 + k3

    loc = check_location(X, Y, defined_domain, x4)[0]
    if loc != "IN":
        return None, None

    # Compute derivative at the end of the time-step.
    x_prime = _scaling_vectorfield_incompressible(X, Y, x4, x_prime, vector_field, interp_eig)

    # x_prime can be None at the boundaries of the spatial domain of the data --> stop integration
    if x_prime is None:
        return None, None

    # compute derivative
    k4 = step_size * x_prime

    # define list for derivatives and positions of particle
    x_prime_update = []
    x_update = []

    # Compute RK4-derivative
    for j in range(2):
        x_prime_update.append(1.0 / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / step_size)

    # Integration x <-- x + x_prime*step_size
    for j in range(2):
        # Update position of particles
        x_update.append(x[j] + x_prime_update[j] * step_size)

    # transform list to arrays
    x_update = np.array(x_update)
    x_prime_update = np.array(x_prime_update)

    # Check if particle is in domain of data
    if check_location(X, Y, defined_domain, x_update)[0] != "IN":
        return None, None

    return x_update, x_prime_update


def _tensorlines_incompressible(X, Y, eig, vector_field, min_distance, max_length, step_size, n_tensorlines=-1,
                                hyperbolicity=0, n_iterations=10 ** 4, verbose=False):
    '''
    Wrapper for RK4_tensorlines_incompressible(). Integrates the tensorlines given an eigenvector/eigenvalue field. The integration stops
    whenever a threshold value is reached (max_length/hyperbolicity/counter_threshold).

    Parameters:
        X:                 array (Ny, Nx), X-meshgrid (of complete data domain).
        Y:                 array (Ny, Nx), Y-meshgrid (of complete data domain).
        eig:               array (Ny, Nx), eigenvalue field.
        vector_field:      array (Ny, Nx, 2), eigenvector field.
        min_distance:      float, minimum distance between local extrema in the eigenvalue field.
        max_length:        float, maximum length of tensorlines.
        step_size:         float, step size used for integration. This value is kept constant.
        n_tensorlines:     int, extract only the most relevant tensorlines. If n_tensorlines = -1, then algorithm returns all possible tensorlines.
        hyperbolicity:     float, threshold on hyperbolicity value.
                           If point on tensorline has hyperbolicity lower than this threshold, than integration is stop.
        n_iterations:      int, threshold on number of iterations.
        verbose:           bool, if True, function reports progress at every 100th iteration

    Returns:
        tensorlines:       list, each element in the list contains an array specifying the x/y-coordinates of the tensorlines.
    '''

    # Find local extrema of eigfield
    peak_x, peak_y, peak_field = _loc_max(min_distance, X, Y, eig)[2:]

    # if you do not place a restriction on the number of tensorlines,
    # then as many tensorlines as possible are plotted
    if n_tensorlines == -1:
        n_tensorlines = len(peak_field)

    # defined domain
    defined_domain = np.isfinite(eig).astype(int)

    # set nan values of eig to zero for gridded interpolation
    eig = np.nan_to_num(eig, 0)

    # Interpolate Eigenvalue field
    interp_eig = RBS(Y[:, 0], X[0, :], eig, kx=1, ky=1)

    # Define list of tensorlines (back/forward)
    tensorlines = [[], []]

    # Iterate over all local maxima
    for i in range(len(peak_x[:n_tensorlines])):

        tensorlines_forw = [[], []]
        tensorlines_back = [[], []]

        # Local maxima point
        x = np.array([peak_x[i], peak_y[i]])

        if peak_field[i] > hyperbolicity:
            # Boolean forward Iteration
            bool_forward, bool_backward = True, True

        else:
            bool_forward, bool_backward = False, False

        # Start integration only if local maxima is not close than 'min_distance' to shrinkline
        if bool_forward and bool_backward:

            # Starting point of integration
            x_forward = x
            x_backward = x

            # Append starting point to list containing positions of forward shrinklines
            for ii in range(2):
                tensorlines_forw[ii].append(x[ii])

            # Check orientation of vector-field and rieorient if needed.
            vx, vy = _orient_vectorfield(X, Y, x, vector_field)

            # Initial vector orientation
            x_prime_forward = np.array([vx, vy])
            x_prime_backward = -np.array([vx, vy])

            # Initial distance
            dist = 0

            # set counter for number of iterations
            counter = 0

            while (bool_forward or bool_backward) and counter < n_iterations:

                if verbose and counter%100==0:
                    print("Percentage completed: ", np.around(counter/n_iterations, int(np.log(n_iterations))+1)*100)

                # Integrate only if 'x_prime_forward' is defined and 'bool_forward == True'
                if bool_forward and x_prime_forward is not None:

                    # RK4 integration for tensorline
                    x_forward, x_prime_forward = _RK4_tensorlines_incompressible(X, Y, defined_domain, x_forward,
                                                                                 x_prime_forward, step_size,
                                                                                 vector_field, interp_eig)

                    if x_forward is not None:

                        # Compute length of tensorline
                        dist += np.sqrt(x_prime_forward[0] ** 2 + x_prime_forward[1] ** 2) * step_size

                        # If distance is below length of tensorline --> append point to tensorline
                        if dist < max_length:
                            for ii in range(2):
                                tensorlines_forw[ii].append(x_forward[ii])

                        else:

                            bool_forward = False

                    else:

                        bool_forward = False

                # Integrate only if 'x_prime_backward' is defined and 'bool_backward == True'
                if bool_backward and x_prime_backward is not None:

                    # RK4 integration for tensorline
                    x_backward, x_prime_backward = _RK4_tensorlines_incompressible(X, Y, defined_domain, x_backward,
                                                                                   x_prime_backward, step_size,
                                                                                   vector_field, interp_eig)

                    if x_backward is not None:

                        # Compute length of tensorline
                        dist += np.sqrt(x_prime_backward[0] ** 2 + x_prime_backward[1] ** 2) * step_size

                        # If distance is below length of tensorline --> append point to tensorline
                        if dist < max_length:
                            for ii in range(2):
                                tensorlines_back[ii].append(x_backward[ii])

                        else:

                            bool_backward = False

                    else:

                        bool_backward = False

                counter += 1

            # Append backward and forward shrinkline
            for ii in range(2):
                tensorlines[ii].append(np.append(np.flip(tensorlines_back[ii]), tensorlines_forw[ii]))

    return tensorlines


# Particle binning function in 2D
# copied from https://stackoverflow.com/questions/61325586/fast-way-to-bin-a-2d-array-in-python
def bin2d(orig_array, bin_width):
    m_bins = orig_array.shape[0]//bin_width
    n_bins = orig_array.shape[1]//bin_width
    return orig_array.reshape(m_bins, bin_width, n_bins, bin_width).sum(3).sum(1)


# Color map function below shared by Lars Larson
def color_change_white(img1, img2, scale=1.0):
    img1 = img1 ** scale
    img2 = img2 ** scale
    img1[img1 > 1] = 1
    img1[img1 < 0] = 0
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0
    comb_img = np.zeros((*img1.shape, 3))

    # Red
    comb_img[:, :, 0] = 1 - img1
    # Blue
    comb_img[:, :, 2] = 1 - img2
    # Green
    comb_img[:, :, 1] = 1 - (img1 + img2)

    comb_img[:, :, 1] = np.clip(comb_img[:, :, 1], 0, 1)  # Clip the green channel

    comb_img = np.real(comb_img)
    comb_img[comb_img > 1] = 1

    return comb_img

