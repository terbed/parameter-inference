import numpy as np
import prior
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood
import plot


class RandomVariable:
    """
    A class representing  random variables
    """

    def __init__(self, name, range_min, range_max, resolution, mean, sigma,  value=None):
        self.name = name
        self.unit = self.get_unit()
        self.value = value
        self.range_min = range_min
        self.range_max = range_max
        self.resolution = resolution
        self.mean = mean
        self.sigma = sigma

        self.step = np.abs(range_max-range_min)/resolution
        self.values = np.linspace(range_min, range_max, resolution)
        self.prior = prior.normal(self.values, mean, sigma)

        self.posterior = []
        self.likelihood = []

        if self.value is None:
            self.value = self.mean

    def get_unit(self):
        database = {'Ra': '[ohm cm]', 'cm': '[uF/cm^2]', 'gpas': '[uS/cm^2]'}
        return database[self.name]


class ParameterSet:
    """
    Takes RandomVariable object and create parameter set sequence, joint distribution and joint step for further use
    """
    def __init__(self, *params):
        self.params = params
        self.name = self.get_name()
        self.parameter_set_seq, self.shape = self.parameter_seq_for_mapping()
        self.joint_prior = prior.normal_nd(*params)
        self.joint_step = self.joint_step()
        self.margin_ax = self.get_margin_ax()
        self.margin_step = self.get_margin_step()
        self.isBatch = False
        self.batch_len = 500
        if len(self.parameter_set_seq) >= self.batch_len*2:
            self.isBatch = True
            self.parameter_set_batch_list = []
            self.create_batch()

    def get_name(self):
        name = ''
        for item in self.params:
            name += item.name
        return name

    def parameter_seq_for_mapping(self):
        """
        Create a set of parameters for multiprocess mapping function

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
        """Create batches with self.batch_len elements for flawless multiprocessing"""
        batch_num = len(self.parameter_set_seq)/self.batch_len  # This stores an int
        for times in range(batch_num):
            self.parameter_set_batch_list.append(self.parameter_set_seq[times*self.batch_len: (times+1)*self.batch_len])

        if len(self.parameter_set_batch_list) < float(len(self.parameter_set_seq))/float(self.batch_len):
            self.parameter_set_batch_list.append(self.parameter_set_seq[batch_num*self.batch_len:])

    def joint_step(self):
        """
        Takes RandomVariable objects and compute there joint step
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
    def __init__(self, target_trace, parameter_set, working_path=''):
        self.p = parameter_set
        self.target = target_trace
        self.working_path = working_path
        self.likelihood = []
        self.posterior = []

    # Method to override
    def run_sim(self, sim_protocol_func, covmat):
        pass

    def run_evaluation(self):
        self.__create_likelihood()
        self.__create_posterior()
        self.__marginalize()

    def __create_likelihood(self):
        self.likelihood = np.reshape(self.likelihood, self.p.shape)
        self.likelihood = np.subtract(self.likelihood, np.amax(self.likelihood))
        self.likelihood = np.exp(self.likelihood)

    def __create_posterior(self):
        self.posterior = np.multiply(self.likelihood, self.p.joint_prior)
        self.posterior /= np.sum(self.posterior) * self.p.joint_step

    def __marginalize(self):
        for idx, item in enumerate(self.p.margin_ax):
            self.p.params[idx].likelihood = \
                np.sum(self.likelihood, axis=tuple(item)) * self.p.margin_step[idx]

            self.p.params[idx].posterior = \
                np.sum(self.posterior, axis=tuple(item)) * self.p.margin_step[idx]

    def __str__(self):
        for item in self.p.params:
            plot.marginal_plot(item, path=self.working_path)
        return "Marginal Plot Done!"


class IndependentInference(Inference):
    def __init__(self, target_trace,  parameter_set, working_path=''):
        Inference.__init__(self, target_trace=target_trace, parameter_set=parameter_set, working_path=working_path)

    def run_sim(self, sim_protocol_func, noise_sigma):
        pool = Pool(multiprocessing.cpu_count()-1)

        # Create mappable functions
        dev_func = partial(likelihood.deviation, model_func=sim_protocol_func, target_trace=self.target)
        log_likelihood_func = partial(likelihood.independent_log_likelihood, noise_sigma=noise_sigma)

        if self.p.isBatch:
            for idx, batch in enumerate(self.p.parameter_set_batch_list):
                print str(idx) + ' batch of work is done out of ' \
                      + str(len(self.p.parameter_set_batch_list))

                batch_likelihood = pool.map(dev_func, batch)
                batch_likelihood = map(log_likelihood_func, batch_likelihood)
                self.likelihood.extend(batch_likelihood)

            pool.close()
            print "log_likelihood: Done!"
        else:
            pool = Pool(multiprocessing.cpu_count())
            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations on all cores..."
            func = partial(likelihood.independent_log_likelihood,
                           model_func=sim_protocol_func,
                           target_trace=self.target,
                           noise_sigma=noise_sigma)

            self.likelihood = pool.map(func, self.p.parameter_set_seq)
            pool.close()
            pool.join()

            print "Done with Simulations!"


class DependentInference(Inference):
    def __init__(self, target_trace,  parameter_set, working_path=''):
        Inference.__init__(self, target_trace=target_trace, parameter_set=parameter_set, working_path=working_path)

    def run_sim(self, sim_protocol_func, inv_covmat):
        print "Run simulations..."
        pool = Pool(multiprocessing.cpu_count())

        dev_func = partial(likelihood.deviation, model_func=sim_protocol_func, target_trace=self.target)
        log_likelihood_func = partial(likelihood.log_likelihood, inv_covmat=inv_covmat)

        if self.p.isBatch:
            for idx, batch in enumerate(self.p.parameter_set_batch_list):
                print str(idx + 1) + ' batch of work is done out of ' \
                      + str(len(self.p.parameter_set_batch_list))

                batch_likelihood = pool.map(dev_func, batch)
                batch_likelihood = map(log_likelihood_func, batch_likelihood)
                self.likelihood.extend(batch_likelihood)

            pool.close()
            print "log_likelihood: Done!"
        else:
            pool = Pool(multiprocessing.cpu_count())
            self.likelihood = pool.map(partial(likelihood.deviation,
                                               model_func=sim_protocol_func,
                                               target_trace=self.target), self.p.parameter_set_seq)

            print "Create likelihood..."
            self.likelihood = map(partial(likelihood.log_likelihood, inv_covmat=inv_covmat), self.likelihood)
            pool.close()
            pool.join()




