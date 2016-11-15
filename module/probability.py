import numpy as np
import prior
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood
from matplotlib import pyplot as plt
import module.plot
import os


class RandomVariable:
    """
    A class representing  random variables
    """

    def __init__(self, name, range_min, range_max, resolution, mean, sigma,  value=None):
        self.name = name
        self.unit = module.plot.return_unit(name)
        self.value = value
        self.range_min = range_min
        self.range_max = range_max
        self.resolution = resolution
        self.mean = mean
        self.sigma = sigma

        self.step = np.abs(range_max-range_min)/resolution
        self.values = np.linspace(range_min, range_max, resolution)
        self.prior = prior.normal(mean, sigma, self.values)

        self.posterior = []
        self.likelihood = []

        if self.value is None:
            self.value = self.mean


class ParameterSet:
    """
    Takes RandomVariable object and create parameter set sequence, joint distribution and joint step for further use
    """
    def __init__(self, *params):
        self.params = params
        self.name = self.set_name()
        self.parameter_set_seq, self.shape = self.parameter_seq_for_mapping()
        self.joint_prior = prior.normal_nd()
        self.joint_step = self.joint_step()
        self.margin_ax = self.get_margin_ax()
        self.margin_step = self.get_margin_step()
        self.isBatch = False
        self.batch_len = 500
        if len(self.parameter_set_seq) >= self.batch_len*2:
            self.isBatch = True
            self.parameter_set_batch_list = []
            self.create_batch()

    def set_name(self):
        name = ''
        for item in self.params:
            name += item.name
        return name

    def parameter_seq_for_mapping(self):
        """
        Create a set of parameters for multiprocess mapping function

        :param params: a tuple of RandomVariable object
        :return: a parameter set sequence dict for multiprocessing and the shape for reshaping the list correctly
                 form: [{'Ra' = 100, 'gpas' = 0.0001, 'cm' = 1}, {'Ra' = 100, 'gpas' = 0.0001, 'cm' = 1.1}, ... ], [200,200,200]
        """
        shape = []
        param_seq = []

        for idx, item in enumerate(self.params):
            shape.append(len(item.values))

        # Create parameter set for multiprocess simulation
        for idx, _ in np.ndenumerate(np.zeros(shape)):
            param_set = {}
            for ax, element in enumerate(idx):
                param_set[self.params[ax].name] = self.params[ax].values[element]
            param_seq.append(param_set)
            del param_set

        return param_seq, shape

    def create_batch(self):
        """Create batches with 50 elements for flawless multiprocessing"""
        batch_num = len(self.parameter_set_seq)/self.batch_len  # This stores an int
        for times in range(batch_num):
            self.parameter_set_batch_list.append(self.parameter_set_seq[times*self.batch_len: (times+1)*self.batch_len])

        if len(self.parameter_set_batch_list) < float(len(self.parameter_set_seq))/float(self.batch_len):
            self.parameter_set_batch_list.append(self.parameter_set_seq[batch_num*self.batch_len:])

    def joint_step(self):
        """
        Takes RandomVariable objects and compute there joint step
        :param params: RandomVariable object
        :return: Joint step (float value)
        """
        step = 1
        for item in self.params:
            step *= item.step

        return step

    def get_margin_ax(self):
        all_ax = range(len(self.params))
        margin_ax = []
        for idx, _ in enumerate(self.params):
            del all_ax[idx]
            margin_ax.append(all_ax)
            all_ax = range(len(self.params))
        return margin_ax

    def get_margin_step(self):
        margin_step = np.ones(len(self.margin_ax))
        for idx, l in enumerate(self.margin_ax):
            for item in l:
                margin_step[idx] *= self.params[item].step
        return margin_step


class Inference:
    """
    Takes a ParameterSet object
    """
    def __init__(self, target_trace, parameter_set):
        self.parameter_set = parameter_set
        self.target = target_trace
        self.likelihood = []
        self.posterior = []

    def run_sim(self, sim_protocol_func, covmat):
        pass

    def run_evaluation(self):
        self.__create_likelihood()
        self.__create_posterior()

    def __create_likelihood(self):
        self.likelihood = np.reshape(self.likelihood, self.parameter_set.shape)
        self.likelihood = np.subtract(self.likelihood, np.amax(self.likelihood))
        self.likelihood = np.exp(self.likelihood)

    def __create_posterior(self):
        self.posterior = np.multiply(self.likelihood, self.parameter_set.joint_prior)
        self.posterior = self.posterior / (np.sum(self.posterior) * self.parameter_set.joint_step)

    def __str__(self):
        # Marginalize likelihood and posterior
        for idx, item in enumerate(self.parameter_set.margin_ax):
            self.parameter_set.params[idx].likelihood = \
                np.sum(self.likelihood, axis=tuple(item)) * self.parameter_set.margin_step[idx]

            self.parameter_set.params[idx].posterior = \
                np.sum(self.posterior, axis=tuple(item)) * self.parameter_set.margin_step[idx]

        for item in self.parameter_set.params:
            # Plot posterior
            plt.figure()
            plt.title(item.name + " posterior (g) and prior (b) distribution")
            plt.xlabel(item.name + ' ' + item.unit)
            plt.ylabel("probability")
            plt.plot(item.values, item.posterior, '#34A52F')
            plt.plot(item.values, item.prior, color='#2FA5A0')

            filename = "/Users/Dani/TDK/parameter_estim/exp/out/" + \
                       self.parameter_set.name + '-'+ item.name + "-posterior_" + str(item.resolution) + "_"
            i = 0
            while os.path.exists('{}{:d}.png'.format(filename, i)):
                i += 1
            plt.savefig('{}{:d}.png'.format(filename, i))
            print "Plot done! File path: " + filename

            # Plot likelihood
            plt.figure()
            plt.title(item.name + " likelihood (r) and prior (b) distribution")
            plt.xlabel(item.name + ' ' + item.unit)
            plt.ylabel("probability")
            plt.plot(item.values, item.likelihood, color='#A52F34')
            plt.plot(item.values, item.prior, color='#2FA5A0')

            filename = "/Users/Dani/TDK/parameter_estim/exp/out/" +\
                       self.parameter_set.name + '-' +item.name + "-likelihood_" + str(item.resolution) + "_"
            i = 0
            while os.path.exists('{}{:d}.png'.format(filename, i)):
                i += 1
            plt.savefig('{}{:d}.png'.format(filename, i))
            print "Plot done! File path: " + filename

        return "Plot Done!"


class IndependentInference(Inference):
    def __int__(self, target_trace,  parameter_set):
        Inference.__init__(target_trace=target_trace, parameter_set=parameter_set)

    def run_sim(self, sim_protocol_func, noise_sigma):
        pool = Pool(multiprocessing.cpu_count())
        self.likelihood = pool.map(partial(likelihood.independent_log_likelihood,
                                           model_func=sim_protocol_func,
                                           target_trace=self.target,
                                           noise_sigma=noise_sigma), self.parameter_set.parameter_set_seq)
        pool.close()
        pool.join()


class DependentInference(Inference):
    def __int__(self, target_trace,  parameter_set):
        Inference.__init__(target_trace=target_trace, parameter_set=parameter_set)

    def run_sim(self, sim_protocol_func, inv_covmat):
        print "Run simulations..."
        pool = Pool(multiprocessing.cpu_count())

        if self.parameter_set.isBatch:
            for idx, batch in enumerate(self.parameter_set.parameter_set_batch_list):
                print str(idx + 1) + ' batch of work is done out of ' + str(len(self.parameter_set.parameter_set_batch_list))

                batch_likelihood = pool.map(partial(likelihood.deviation,
                                                    model_func=sim_protocol_func,
                                                    target_trace=self.target), batch)

                batch_likelihood = map(partial(likelihood.log_likelihood, inv_covmat=inv_covmat), batch_likelihood)

                self.likelihood.extend(batch_likelihood)
            print "log_likelihood: Done!"
        else:
            self.likelihood = pool.map(partial(likelihood.deviation,
                                               model_func=sim_protocol_func,
                                               target_trace=self.target), self.parameter_set.parameter_set_seq)

            print "Create likelihood..."
            self.likelihood = map(partial(likelihood.log_likelihood, inv_covmat=inv_covmat), self.likelihood)

        pool.close()
        pool.join()


