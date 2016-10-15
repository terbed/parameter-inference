import numpy as np


def normal_val(mean, sigma, x):
    """Returns a value for the given x"""
    normal_distribution_value = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
    return normal_distribution_value


def normal(mean, sigma, values):
    """Returns a normal distribution as a numpy array"""
    prior = []
    for x in values:
        prior.append(normal_val(mean, sigma, x))

    return np.array(prior)


def normal2d(x_mean, x_sigma, x_values, y_mean, y_sigma, y_values):
    """Returns a two dimensional normal distribution as a numpy array"""
    prior = np.zeros((len(x_values), len(y_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            prior[i, j] = normal_val(x_mean, x_sigma, x) * normal_val(y_mean, y_sigma, y)

    return prior
