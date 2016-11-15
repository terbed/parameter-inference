import numpy as np


def independent_1d(simulation_func, x_values, y, sigma, target_trace):
    """Returns the one dimensional likelihood function for the INDEPENDENT observation as a numpy array

    simulation_func: the set-up function for the given simulation
    x_values: sampled variable array
    target_trace: the experimental (noisy) trace array
    """

    log_likelihood = []
    for i, x in enumerate(x_values):
        (_, v) = simulation_func(x, y)
        v_dev = np.subtract(target_trace, v)
        loglival = - np.sum(np.square(v_dev)) / (2 * sigma ** 2)
        log_likelihood.append(loglival)

    log_likelihood = np.subtract(log_likelihood, np.amax(log_likelihood))

    return np.exp(log_likelihood)


def independent_2d(simulation_func, x_values, y_values, sigma, target_trace):
    """Returns the two dimensional likelihood function for the INDEPENDENT observation as a numpy array

    simulation_func: the set-up function for the given simulation
    x_values: sampled variable array
    y_values: sampled variable array
    target_trace: the experimental (noisy) trace array
    """
    log_likelihood = np.zeros((len(x_values), len(y_values)))

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            (_, v) = simulation_func(x, y)
            v_dev = np.subtract(target_trace, v)
            log_likelihood[i, j] = - np.sum(np.square(v_dev)) / (2 * sigma ** 2)

    log_likelihood = np.subtract(log_likelihood, np.amax(log_likelihood))

    return np.exp(log_likelihood)


def dependent_2d(simulation_func, x_values, y_values, inv_covmat, target_trace):
    """Returns the two dimensional likelihood function for the DEPENDENT observation as a numpy array

    simulation_func: set-up function for the given simulation
    x_values: sampled variable array
    y_values: sampled variable array
    inv_covmat: the inverse of the covariance matrix,
    target_trace: the experimental (noisy) trace array
    """
    log_likelihood = np.zeros((len(x_values), len(y_values)))

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            (_, v) = simulation_func(x, y)
            v_dev = np.array(np.subtract(target_trace, v))
            log_likelihood[i, j] = - 1 / 2 * np.dot(v_dev, np.array(inv_covmat).dot(v_dev))

    log_likelihood = np.subtract(log_likelihood, np.amax(log_likelihood))

    return np.exp(log_likelihood)


# For multiprocessing -------------------------------------------------------------------------------------------


def independent_log_likelihood(param_set, model_func, target_trace, noise_sigma):
    (_, v) = model_func(**param_set)
    v_dev = np.subtract(target_trace, v)
    return - np.sum(np.square(v_dev)) / (2 * noise_sigma ** 2)


def deviation(param_set, model_func, target_trace):
    (_, v) = model_func(**param_set)
    v_dev = np.array(np.subtract(target_trace, v))
    return v_dev


def log_likelihood(dev, inv_covmat):
    return - 1 / 2 * np.inner(dev, inv_covmat.dot(dev))


