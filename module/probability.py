import numpy as np
import prior
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood
from module.trace import analyse, kl_test
import plot
from save_load import save_file, save_params
import os


class RandomVariable:
    """
    A class representing  random variables
    """

    def __init__(self, name, range_min, range_max, resolution, mean, sigma,  value=None, p_sampling='u'):
        """

        :param name: the name of the parameter (string)
        :param range_min: minimum simulation range
        :param range_max: maximum simulation range
        :param resolution: resolution of simulation
        :param mean: the mean of the prior distribution
        :param sigma: standard deviation of the prior distribution
        :param value: the true value (optional) if not given, then value=mean
        :param p_sampling: parameter sampling method: uniform: 'u' or prior: 'p' (sampled from prior distribution)
        """

        self.name = name
        self.unit = self.get_unit()
        self.mean = mean
        self.value = value
        if self.value is None:
            self.value = self.mean
        self.range_min = range_min
        self.range_max = range_max
        self.offset = self.get_offset()
        self.resolution = resolution
        self.sampling_type = p_sampling
        self.sigma = sigma
        self.step = np.abs(range_max-range_min)/resolution
        self.values = self.__get_values()

        self.prior = prior.normal(self.values, mean, sigma)
        self.likelihood = []
        self.posterior = []
        self.fitted_gauss = []

        # Maximum inferred parameter values
        self.max_p = None
        self.max_marginal_p = None
        self.max_l = None
        self.max_marginal_l = None

        if self.value is None:
            self.value = self.mean

    def get_unit(self):
        database = {'Ra': '[ohm cm]',
                    'cm': '[uF/cm^2]',
                    'gpas': '[uS/cm^2]',
                    'gpas_soma': '[uS/cm^2]',
                    'k': '1/(um)',
                    'ffact': ''}
        return database[self.name]

    def get_init(self):
        return [self.name, str(self.range_min), str(self.range_max),
                str(self.resolution), str(self.mean), str(self.sigma), str(self.value)]

    # The typical range around mean fo parameter
    def get_offset(self):
        database = {'Ra': 50., 'cm': 0.5, 'gpas': 0.00005, 'gpas_soma': 0.00005, 'k': 0.0005, 'ffact': 2.5}
        return database[self.name]

    def __get_values(self):
        if self.sampling_type == 'u':
            return np.linspace(self.range_min, self.range_max, self.resolution)
        else:
            values = np.random.normal(self.mean, self.sigma, self.resolution)
            values = np.sort(values)
            return values


class ParameterSet:
    """
    Takes RandomVariable object and create parameter set sequence, joint distribution and joint step for further use
    """
    def __init__(self, *params):
        self.params = params
        self.name = self.get_name()
        self.optimized_params = self.get_optimized_params()  # This dictionary contains the prior optimized params
        self.parameter_set_seq, self.shape = self.parameter_seq_for_mapping()
        self.joint_prior = prior.normal_nd(*params)
        self.joint_step = self.joint_step()
        self.margin_ax = self.get_margin_ax()
        self.margin_step = self.get_margin_step()
        self.isBatch = False
        self.batch_len = 30000
        self.parameter_set_batch_list = []

    def get_name(self):
        name = ''
        for item in self.params:
            name = name + '-' + item.name
        return name

    def get_optimized_params(self):
        opt_dict = {}
        for item in self.params:
            opt_dict[item.name] = item.value
        return opt_dict

    def parameter_seq_for_mapping(self):
        """
        Create a set of parameters for multiprocess mapping function

        :return: a parameter set sequence dict for multiprocessing and the shape for reshaping the list correctly
                 form: [{'Ra' = 100, 'gpas' = 0.0001, 'cm' = 1}, {'Ra' = 100, 'gpas' = 0.0001, 'cm' = 1.1}, ... ], [200,200,200]
        """

        shape = []
        param_seq = []

        for item in self.params:
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
        print "Creating batches for flawless simulations..."
        batch_num = len(self.parameter_set_seq)/self.batch_len  # This stores an int
        for times in range(batch_num):
            self.parameter_set_batch_list.append(self.parameter_set_seq[times*self.batch_len: (times+1)*self.batch_len])

        # Check if there is left over
        if float(len(self.parameter_set_batch_list)) < len(self.parameter_set_seq)/float(self.batch_len):
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
        """
        :return: Gives back for all parameters all the other ones:
                For example: [[1,2], [0,2], [0,1]]
        """
        all_ax = range(len(self.params))
        margin_ax = []
        for idx, _ in enumerate(self.params):
            del all_ax[idx]
            margin_ax.append(all_ax)
            all_ax = range(len(self.params))
        return margin_ax

    def get_margin_step(self):
        """
        :return: Margin step for margin ax:
                For example: [0.3, 234.23, 0.0034]
        """
        margin_step = np.ones(len(self.margin_ax))
        for idx, l in enumerate(self.margin_ax):
            for item in l:
                margin_step[idx] *= self.params[item].step
        return margin_step


class Inference:
    """
    Takes a ParameterSet object
    """
    def __init__(self, model, target_trace, parameter_set, working_path='', save=True, speed='max'):
        """
        :param model: model function 
        :param target_trace: Experimental (or synthetic) data
        :param parameter_set: ParameterSet object
        :param noise: Noise sigma or inverse covariant matrix
        :param working_path: The result will be saved here
        :param debugging: 
        :param speed: 'max', 'mid', 'min' or single multiprocessing options. In some system only the 'single' works...
        """
        self.m = model
        self.p = parameter_set
        self.target = target_trace
        self.working_path = working_path
        self.check_directory(working_path)
        self.save = save
        self.speed = speed

        self.likelihood = []
        self.posterior = []
        self.KL = 0.

    @staticmethod
    def check_directory(working_path):
        if not os.path.exists(working_path):
            os.makedirs(working_path)

    # Method to override
    def run_sim(self):
        pass

    def run_moretrace_inf(self):
        pass

    def run_evaluation(self):
        print "\nRunning evaluation of the result..."
        if self.save:
            self.__save_result()

        self.__create_likelihood()
        self.__create_posterior()
        self.__marginalize()
        self.__max_probability()
        self.__fit_posterior()

        print "Check FULL posterior correctness: the integrate of posterior: + " + str(np.sum(self.posterior)*self.p.joint_step)

    def __create_likelihood(self):
        self.likelihood = np.reshape(self.likelihood, self.p.shape)
        self.likelihood = np.subtract(self.likelihood, np.amax(self.likelihood))
        self.likelihood = np.exp(self.likelihood)

    def __create_posterior(self):
        self.posterior = np.multiply(self.likelihood, self.p.joint_prior)
        self.posterior /= np.sum(self.posterior) * self.p.joint_step
        self.KL = kl_test(self.posterior.flat, self.p.joint_prior.flat, self.p.joint_step)

    def __max_probability(self):
        max_l = self.p.parameter_set_seq[np.argmax(self.likelihood)]
        max_p = self.p.parameter_set_seq[np.argmax(self.posterior)]

        for idx, item in enumerate(self.p.params):
            item.max_l = max_l[item.name]
            item.max_marginal_l = self.p.params[idx].values[np.argmax(self.p.params[idx].likelihood)]

            item.max_p = max_p[item.name]
            item.max_marginal_p = self.p.params[idx].values[np.argmax(self.p.params[idx].posterior)]

    def __marginalize(self):
        for idx, item in enumerate(self.p.margin_ax):
            self.p.params[idx].likelihood = \
                np.sum(self.likelihood, axis=tuple(item)) * self.p.margin_step[idx]

            self.p.params[idx].posterior = \
                np.sum(self.posterior, axis=tuple(item)) * self.p.margin_step[idx]

    def __fit_posterior(self):
        from module.trace import fit_normal
        for item in self.p.params:
            item.fitted_gauss = fit_normal(item.values, item.posterior, item.value, item.sigma)

    def __save_result(self):
        save_file(self.likelihood, self.working_path + "/loglikelihood", "loglikelihood", header=str(self.p.name) + str(self.p.shape))
        save_params(self.p.params, path=self.working_path + "/loglikelihood")
        save_file(self.target, self.working_path + "/loglikelihood", "target_trace")
        print "loglikelihood.txt data Saved!"

    def __str__(self):
        for item in self.p.params:
            plot.marginal_plot(item, path=self.working_path)
        return "Marginal Plot Done!"

    def analyse_result(self):
        """
        :return: (fitted_sigma, fit_err, relative_deviation, acc, sharper, broader) tuple
        """
        print "\n Running analysation..."

        # Do some analysis on results
        info = []
        for item in self.p.params:
            if item.fitted_gauss[0][0] is not None:
                info.append(analyse(item, item.fitted_gauss))
            else:
                return None

        return info


class IndependentInference(Inference):
    def __init__(self, model, noise_std, target_trace,  parameter_set, working_path='',speed='max', save=True):
        Inference.__init__(self, target_trace=target_trace, parameter_set=parameter_set,
                           working_path=working_path,speed=speed, model=model, save=save)
        self.std = noise_std

    def run_sim(self):
        """
        Run Single simulation
        :return: 
        """

        if self.speed == "min":
            if self.p.isBatch:
                pool = Pool(multiprocessing.cpu_count() - 1)

                # Create mappable functions
                dev_func = partial(likelihood.deviation, model_func=self.m, target_trace=self.target)
                log_likelihood_func = partial(likelihood.independent_log_likelihood, noise_sigma=self.std)

                for idx, batch in enumerate(self.p.parameter_set_batch_list):
                    print str(idx) + ' batch of work is done out of ' \
                          + str(len(self.p.parameter_set_batch_list))
    
                    batch_likelihood = pool.map(dev_func, batch)
                    batch_likelihood = map(log_likelihood_func, batch_likelihood)
                    self.likelihood.extend(batch_likelihood)
    
                pool.close()
                pool.join()
                print "log_likelihood: Done!"
            else:
                pool = Pool(multiprocessing.cpu_count()-1)
    
                # Create mappable functions
                dev_func = partial(likelihood.deviation, model_func=self.m, target_trace=self.target)
                log_likelihood_func = partial(likelihood.independent_log_likelihood, noise_sigma=self.std)
    
                print "Running " + str(len(self.p.parameter_set_seq)) + " simulations on all cores..."
    
                self.likelihood = pool.map(dev_func, self.p.parameter_set_seq)
                self.likelihood = map(log_likelihood_func, self.likelihood)
                pool.close()
                pool.join()
    
                print "Done with Simulations!"
        elif self.speed == 'mid':
            pool = Pool(multiprocessing.cpu_count() - 1)

            # Create mappable functions
            dev_func = partial(likelihood.deviation, model_func=self.m, target_trace=self.target)
            log_likelihood_func = partial(likelihood.independent_log_likelihood, noise_sigma=self.std)

            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations on all cores..."

            self.likelihood = pool.map(dev_func, self.p.parameter_set_seq)
            self.likelihood = map(log_likelihood_func, self.likelihood)
            pool.close()
            pool.join()
        elif self.speed == 'single':
            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations..."
            log_likelihood_func = partial(likelihood.ill, model=self.m, target_trace=self.target, noise_sigma=self.std)
            self.likelihood = map(log_likelihood_func, self.p.parameter_set_seq)
        else:
            pool = Pool(multiprocessing.cpu_count() - 1)
            log_likelihood_func = partial(likelihood.ill, model=self.m, target_trace=self.target, noise_sigma=self.std)
            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations on all cores..."

            self.likelihood = pool.map(log_likelihood_func, self.p.parameter_set_seq)
            pool.close()
            pool.join()

            print "log likelihood DONE!"


class DependentInference(Inference):
    def __init__(self, model, invcovmat, target_trace,  parameter_set, working_path='',speed='max',save=True):
        Inference.__init__(self, target_trace=target_trace, parameter_set=parameter_set,
                           working_path=working_path, speed=speed, save=save, model=model)
        self.invcovmat = invcovmat

    def run_sim(self):
        print "Run simulations..."

        if self.speed == "min":
            if self.p.isBatch:
                pool = Pool(multiprocessing.cpu_count())

                dev_func = partial(likelihood.deviation, model_func=self.m, target_trace=self.target)
                log_likelihood_func = partial(likelihood.log_likelihood, inv_covmat=self.invcovmat)

                for idx, batch in enumerate(self.p.parameter_set_batch_list):
                    print str(idx + 1) + ' batch of work is done out of ' \
                          + str(len(self.p.parameter_set_batch_list))

                    batch_likelihood = pool.map(dev_func, batch)

                    # Compute the exponent with map function (pool.map would crash for this...)
                    batch_likelihood = map(log_likelihood_func, batch_likelihood)
                    self.likelihood.extend(batch_likelihood)

                pool.close()
                pool.join()
            else:
                print "Simulation running..."
                pool = Pool(multiprocessing.cpu_count()-1)

                dev_func = partial(likelihood.deviation, model_func=self.m, target_trace=self.target)
                log_likelihood_func = partial(likelihood.log_likelihood, inv_covmat=self.invcovmat)

                self.likelihood = pool.map(dev_func, self.p.parameter_set_seq)
                self.likelihood = map(log_likelihood_func, self.likelihood)
                pool.close()
                pool.join()
        elif self.speed == "mid":
            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations..."
            pool = Pool(multiprocessing.cpu_count() - 1)

            dev_func = partial(likelihood.deviation, model_func=self.m, target_trace=self.target)
            log_likelihood_func = partial(likelihood.log_likelihood, inv_covmat=self.invcovmat)

            self.likelihood = pool.map(dev_func, self.p.parameter_set_seq)
            self.likelihood = map(log_likelihood_func, self.likelihood)
            pool.close()
            pool.join()
        elif self.speed == "single":
            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations..."
            log_likelihood_func = partial(likelihood.ll, model=self.m, target_trace=self.target,
                                          inv_covmat=self.invcovmat)
            self.likelihood = map(log_likelihood_func, self.p.parameter_set_seq)
        else:
            pool = Pool(multiprocessing.cpu_count() - 1)
            log_likelihood_func = partial(likelihood.ll, model=self.m, target_trace=self.target,
                                          inv_covmat=self.invcovmat)
            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations on all cores..."

            self.likelihood = pool.map(log_likelihood_func, self.p.parameter_set_seq)
            pool.close()
            pool.join()

            print "log likelihood DONE!"




