import numpy as np
import prior
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood


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

"""
class Inference:
    def __init__(self, *parameter_set):
        self.parameters = parameter_set
        self.param_seq = []
        self.shape = []
        self.likelihood = []
        self.posterior = []

        for idx, item in enumerate(parameter_set):
            self.shape.append(len(item.values))

        # Create parameter set for multiprocess simulation
        for idx, _ in np.ndenumerate(np.zeros(self.shape)):
            param_set = []
            for ax, element in enumerate(idx):
                param_set.append(parameter_set[ax].values[element])
            self.param_seq.append(param_set)
            del param_set

    def run_inference(self, sim_protocol_func):
        pass


class IndependentInference(Inference):
    def __int__(self, noise_sigma,  target_trace,  params):
        super(IndependentInference, self).__init__(*params)
        self.sigma = noise_sigma
        self.target = target_trace

    def run_inference(self, sim_protocol_func):
        if __name__ == '__main__':
            pool = Pool(multiprocessing.cpu_count())
            self.likelihood = pool.map(partial(likelihood.independent_log_likelihood,
                                               model_func=sim_protocol_func, target_trace=self.target), self.param_seq)
            pool.close()
            pool.join()

        self.likelihood = np.reshape(self.likelihood, self.shape)
        self.likelihood = np.subtract(self.likelihood, np.amax(self.likelihood))
        self.likelihood = np.exp(self.likelihood)

        self.posterior = np.multiply(self.likelihood, prior.normal_nd(self.parameters.prior))
        self.posterior = self.posterior / (np.sum(self.posterior) * self.parameters.step


class DependentInference(Inference):
    def __int__(self, inv_covmat,  target_trace,  params):
        super(DependentInference, self).__init__(*params)
        self.invcovmat = inv_covmat
        self.target = target_trace

    def run_inference(self, sim_protocol_func):
        pass
"""