import numpy as np
from matplotlib import pyplot as plt

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


def independent_log_likelihood(dev, noise_sigma):
    return - np.sum(np.square(dev)) / (2 * noise_sigma ** 2)


def deviation(param_set, model_func, target_trace):
    (_, v) = model_func(**param_set)  # ** -> to pass dictionary for function variables
    v_dev = np.subtract(target_trace, v)
    return v_dev


def log_likelihood(dev, inv_covmat):
    return - 1 / 2 * np.inner(dev, inv_covmat.dot(dev))


def ll(param_set, model, target_trace, inv_covmat):
    (_, v) = model(**param_set)
    v = np.subtract(target_trace, v)
    return - 1 / 2 * np.inner(v, inv_covmat.dot(v))


def ill(param_set, model, target_trace, noise_sigma):
    (_, v) = model(**param_set)
    v = np.subtract(target_trace, v)
    return - np.sum(np.square(v)) / (2 * noise_sigma ** 2)


def mill(param_set, model, target_traces, noise_std):
    """
    MultiIndependentLogLikelihood

    Evaluate more than one target trace at one simulation: for repetition and fixed_params
    :param param_set: 
    :param model: 
    :param target_traces: 
    :param noise_std: 
    :return: (fixed_param_num, noise_repetition_num) shaped list: [ [...], [...], [...],...]
    """

    (_, v) = model(**param_set)
    pnum = target_traces.shape[0]
    rep = target_traces.shape[1]

    log_l = []

    for j in range(pnum):
        current_param = []
        for idx in range(rep):
            dev = np.subtract(target_traces[j, idx, :], v)
            current_param.append(-np.sum(np.square(dev)) / (2 * noise_std ** 2))

        log_l.append(current_param)
        current_param = []

    return log_l


def mdll(param_set, model, target_traces, inv_covmat):
    """
    Multi Dependent LogLikelihood

    :param param_set:
    :param model:
    :param target_traces:
    :param inv_covmat:
    :return:
    """

    (_, v) = model(**param_set)
    pnum = target_traces.shape[0]
    rep = target_traces.shape[1]

    log_l = []

    for j in range(pnum):
        current_param = []
        for idx in range(rep):
            dev = np.subtract(target_traces[j, idx, :], v)
            current_param.append(- 1/2 * np.inner(dev, inv_covmat.dot(dev)))
        log_l.append(current_param)
        current_param = []

    return log_l


def rill(param_set, model, target_traces, noise_std):
    """
    RepetitionIndependentLogLikelihood

    Evaluate more than one target trace at one simulation: only repetition
    :param param_set:
    :param model:
    :param target_traces:
    :param noise_std:
    :return:
    """

    (_, v) = model(**param_set)
    rep = target_traces.shape[0]

    log_l = []

    for idx in range(rep):
        dev = np.subtract(target_traces[idx, :], v)
        log_l.append(-np.sum(np.square(dev)) / (2 * noise_std ** 2))

    return log_l