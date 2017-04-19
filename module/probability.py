import numpy as np
import prior
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood
import plot
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
        :param value: the true value (optional) if not given, than value=mean
        :param p_sampling: parameter sampling method: uniform: 'u' or prior: 'p' (sampled from prior distribution)
        """
        self.name = name
        self.unit = self.get_unit()
        self.value = value
        self.range_min = range_min
        self.range_max = range_max
        self.resolution = resolution
        self.sampling_type = p_sampling
        self.mean = mean
        self.sigma = sigma
        self.step = np.abs(range_max-range_min)/resolution
        self.values = self.__get_values()

        self.prior = prior.normal(self.values, mean, sigma)
        self.posterior = []
        self.likelihood = []

        if self.value is None:
            self.value = self.mean

    def get_unit(self):
        database = {'Ra': '[ohm cm]', 'cm': '[uF/cm^2]', 'gpas': '[uS/cm^2]'}
        return database[self.name]

    def __get_values(self):
        if self.sampling_type == 'u':
            print "\nUniform parameter sampling: " + self.name
            return np.linspace(self.range_min, self.range_max, self.resolution)
        else:
            print "\nParameter sampling from prior distribution: " + self.name
            values = np.random.normal(self.mean, self.sigma, self.resolution)
            values = np.sort(self.values)
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
        self.batch_len = 100
        if len(self.parameter_set_seq) >= self.batch_len*2:
            self.isBatch = True
            self.parameter_set_batch_list = []
            self.create_batch()

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
    def __init__(self, target_trace, parameter_set, working_path='', debugging=False):
        self.p = parameter_set
        self.target = target_trace
        self.working_path = working_path
        self.check_directory(working_path)
        self.toDebug = debugging
        if self.toDebug:
            self.check_directory(working_path + "/debug")

        self.likelihood = []
        self.posterior = []
        if self.toDebug:
            self.deviation = []

    @staticmethod
    def check_directory(working_path):
        if not os.path.exists(working_path):
            os.makedirs(working_path)

    # Method to override
    def run_sim(self, sim_protocol_func, covmat):
        pass

    def run_evaluation(self, model=None):
        if self.toDebug:
            if model is None:
                print "\nError! You are in debugging mode, so you have to give the right model function!"
                exit(17)

            data = np.zeros((len(self.p.parameter_set_seq), 4), dtype=float)
            data[:, 0] = [np.sum(vec)**2 for vec in self.deviation]
            data[:, 1] = self.likelihood
            data[:, 2] = np.subtract(self.likelihood, np.amax(self.likelihood))
            data[:, 3] = np.exp(np.subtract(self.likelihood, np.amax(self.likelihood)))

            # Save out data
            header = self.p.name + "\ndeviation^2\tlog_likelihood\tnormed\tlikelihood"
            plot.save_file(data, self.working_path+"/debug", "data", header)

            # Do some trace plot TODO plot informative cases accordingly to likelihood or deviation...
            from matplotlib import pyplot as plt
            # Chose parameter-sets to analyse uniformly
            idx = np.linspace(0, len(self.p.parameter_set_seq), num=100, dtype=int, endpoint=False)

            for i in idx:
                t, v = model(**self.p.parameter_set_seq[i])
                summed_dev = np.sum(self.deviation[i])

                plt.figure()
                plt.xlabel("Time [ms]")
                plt.ylabel("[mV]")
                plt.title("Current params(b): %s\n"
                          "Summed dev: %.2e | log_L: %.2e"
                          % (str(self.p.parameter_set_seq[i]),
                             float(summed_dev), float(self.likelihood[i])))
                plt.plot(t, self.target, color='#A52F34')
                plt.plot(t, v, color='#2FA5A0')
                filename = self.working_path + "/debug/trace" + str(i)
                i = 0
                while os.path.exists('{}({:d}).png'.format(filename, i)):
                    i += 1
                plt.savefig('{}({:d}).png'.format(filename, i))

        self.__create_likelihood()
        self.__create_posterior()
        self.__marginalize()
        print "\nCheck FULL posterior correctness: the integrate of posterior: + " + str(np.sum(self.posterior)*self.p.joint_step)

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
    def __init__(self, target_trace,  parameter_set, working_path='', debugging=False):
        Inference.__init__(self, target_trace=target_trace, parameter_set=parameter_set, working_path=working_path, debugging=debugging)

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
            # plot.save_file(self.likelihood, self.working_path, "/log_likelihood.txt", header=str(self.p.name) + str(self.p.shape))
            # print "log_likelihood: Saved!"
        else:
            pool = Pool(multiprocessing.cpu_count())

            # Create mappable functions
            dev_func = partial(likelihood.deviation, model_func=sim_protocol_func, target_trace=self.target)
            log_likelihood_func = partial(likelihood.independent_log_likelihood, noise_sigma=noise_sigma)

            print "Running " + str(len(self.p.parameter_set_seq)) + " simulations on all cores..."

            self.likelihood = pool.map(dev_func, self.p.parameter_set_seq)
            self.likelihood = pool.map(log_likelihood_func, self.likelihood)
            pool.close()
            pool.join()

            print "Done with Simulations!"


class DependentInference(Inference):
    def __init__(self, target_trace,  parameter_set, working_path='', debugging=False):
        Inference.__init__(self, target_trace=target_trace, parameter_set=parameter_set, working_path=working_path, debugging=debugging)

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
                # In the case of debugging we save the deviation vector too
                if self.toDebug:
                    self.deviation.extend(batch_likelihood)

                # Compute the exponent with map function (pool.map would crash for this...)
                batch_likelihood = map(log_likelihood_func, batch_likelihood)
                self.likelihood.extend(batch_likelihood)

            pool.close()
            #plot.save_file(self.likelihood, self.working_path, "/log_likelihood.txt", header=str(self.p.name) + str(self.p.shape))
            #print "log_likelihood: Saved!"
        else:
            print "Simulation running..."
            pool = Pool(multiprocessing.cpu_count())

            dev_func = partial(likelihood.deviation, model_func=sim_protocol_func, target_trace=self.target)
            log_likelihood_func = partial(likelihood.log_likelihood, inv_covmat=inv_covmat)

            self.likelihood = pool.map(dev_func, self.p.parameter_set_seq)
            self.likelihood = pool.map(log_likelihood_func, self.likelihood)
            if self.toDebug:
                self.deviation = self.likelihood
            self.likelihood = map(log_likelihood_func, self.likelihood)
            pool.close()
            pool.join()




