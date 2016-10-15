import numpy as np


def independent(simulation_func, x_values, y_values, sigma, target_trace):
    """Returns the two dimensional likelihood function for the INDEPENDENT observation as a numpy array

    simulation_func: the set-up function for the given simulation
    x_values: sampled variable array
    y_values: sampled variable array
    target_trace: the experimental (noisy) trace array
    """
    x_y = np.zeros((len(x_values), len(y_values)))

    # Fill 2D likelihood matrix with log_likelihood elements
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            (_, v) = simulation_func(x, y)
            v_dev = np.subtract(target_trace, v)
            log_likelihood = - np.sum(np.square(v_dev)) / (2 * sigma ** 2)

            x_y[i, j] = log_likelihood

    x_y = np.subtract(x_y, np.amax(x_y))

    return np.exp(x_y)


def dependent(simulation_func, x_values, y_values, inv_covmat, target_trace):
    """Returns the two dimensional likelihood function for the DEPENDENT observation as a numpy array

    simulation_func: set-up function for the given simulation
    x_values: sampled variable array
    y_values: sampled variable array
    inv_covmat: the inverse of the covariance matrix,
    target_trace: the experimental (noisy) trace array
    """
    x_y = np.zeros((len(x_values), len(y_values)))

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            (_, v) = simulation_func(x, y)
            v_dev = np.array(np.subtract(target_trace, v))
            exponent = - 1 / 2 * np.dot(v_dev, np.array(inv_covmat).dot(v_dev))
            x_y[i, j] = exponent

    x_y = np.subtract(x_y, np.amax(x_y))

    return np.exp(x_y)
