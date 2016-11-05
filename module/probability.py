import numpy as np
import prior


class RandomVariable:
    """
    A class representing  random variables
    """

    def __init__(self, range_min, range_max, resolution, mean, sigma, is_target=False, value=None):
        self.value = value
        self.range_min = range_min
        self.range_max = range_max
        self.resolution = resolution
        self.mean = mean
        self.sigma = sigma
        self.isTarget = is_target

        self.step = np.abs(range_max-range_min)/resolution
        self.values = np.linspace(range_min, range_max, resolution)
        self.prior = prior.normal(mean, sigma, self.values)

        if is_target:
            self.posterior = None

        if self.value is None:
            self.value = self.mean


class ParameterSet:

    def __init__(self, *params):
        self.num_of_var = len(params)
        self.set = params
