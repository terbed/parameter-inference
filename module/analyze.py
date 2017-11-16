import numpy as np
from module.trace import analyse, kl_test
import module.plot as plot
import os


class Analyse:
    """
    Takes a ParameterSet object
    """
    def __init__(self, loglikelihood, parameter_set, working_path=''):
        """
        :param loglikelihood: Flat loglikelihood list
        :param parameter_set: ParameterSet object
        :param working_path: The result will be saved here
        """

        self.check_directory(working_path)
        self.p = parameter_set
        self.working_path = working_path
        self.check_directory(working_path)

        self.likelihood = loglikelihood
        self.posterior = []
        self.KL = 0.

        self.run_evaluation()

    @staticmethod
    def check_directory(working_path):
        if not os.path.exists(working_path):
            os.makedirs(working_path)

    def run_evaluation(self):
        print "\nConstructing posterior distribution..."

        self.__create_likelihood()
        self.__create_posterior()
        self.__max_probability()
        self.__marginalize()
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

        for item in self.p.params:
            item.max_l = max_l[item.name]
            item.max_p = max_p[item.name]

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

    def __str__(self):
        # # Plot marginals
        # for item in self.p.params:
        #     plot.marginal_plot(item, path=self.working_path)
        #
        # # Plot joints
        # for item in self.p.margin_ax:
        #     plot.plot_joint(self, self.p.params[item[0]], self.p.params[item[1]])

        # fullplot
        plot.fullplot(self)

        return "Plot Done!\n" \
               "(fitted_sigma, fit_err, relative_deviation, acc, sharper, broader)\n" + str(self.analyse_result())\
               + "\n KLD: " + str(self.KL)

    def marginal_plot(self):
        # Plot marginals
        for item in self.p.params:
            plot.marginal_plot(item, path=self.working_path)

    def joint_plot(self):
        # Plot joints
        for item in self.p.margin_ax:
            plot.plot_joint(self, self.p.params[item[0]], self.p.params[item[1]])

    def analyse_result(self):
        """
        :return: (fitted_sigma, fit_err, relative_deviation, acc, sharper, broader) tuple for each parameter
        """
        print "\n Running analysis on inference result..."

        # Do some analysis on results
        info = []
        for item in self.p.params:
            if item.fitted_gauss[0][0] is not None:
                info.append(analyse(item, item.fitted_gauss))
            else:
                return None

        return info

    def get_broadness(self):
        """
        :return: broader value for each parameter
        """
        print "\n get_broadness()..."

        # Do some analysis on results
        info = []
        for item in self.p.params:
            if item.fitted_gauss[0][0] is not None:
                info.append(analyse(item, item.fitted_gauss)[5])
            else:
                print "--- Cannot fit normal to posterior!!! ---"
                return info.append(100.)

        return info