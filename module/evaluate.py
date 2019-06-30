from module.protocol_test import plot_single_results, plot_combined_results, mult_likelihood,\
    combine_likelihood, protocol_comparison, plot_single_results_reps
import time
import tables as tb
import os


def check_directory(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)


class Evaluation:
    """
    Class for evaluating the result of simulations
    """

    def __init__(self, n_fixed_params, n_rep, p_names, dir, subdirs):
        """

        :param n_fixed_params: number of fixed parameters
        :param n_rep: number of repitition
        :param p_names: list of parameter names \ example: ["gpas", "k"]
        :param dir: simulation output directory
        :param subdirs: subdirectories in the man directory
        :param comb_lists: list of list of protocol names to combine \ struct: [ ["p1", "p2"], ["p4", "p5"] ]
        """

        self.nfp = n_fixed_params
        self.nrep = n_rep
        self.pnames = p_names
        self.rootdir = dir
        self.subdirs = subdirs

        # Load parameter space initializator
        self.pinit = tb.open_file(dir + "paramsetup.hdf5", mode="r")

    def __del__(self):
        self.pinit.close()

    def single_result_plot(self, which_rep=0):
        """
        This function save the plots of likelihoods and posteriors for each fixed parameter and one chosen repetition
        (because it would be too many to plot all of them...)

        :param which_rep: at which repetition number to plot results (should be less than the number of repetitions...)
        :return: save the plots of the single likelihood and posterior distributions at given repnum for all fix params
        """

        for protocol in self.subdirs:
            plot_single_results(path=(self.rootdir + protocol), numfp=self.nfp, which=which_rep, dbs=self.pinit)

    def single_result_plot_allrep(self, which_fp=0):
        """
        This function save the plots of likelihoods and posteriors for one fixed parameter all repetition
        (because it would be too many to plot all of them...)

        :param which_fp: at which repetition number to plot results (should be less than the number of repetitions...)
        :return: save the plots of the single likelihood and posterior distributions at given repnum for all fix params
        """

        for protocol in self.subdirs:
            plot_single_results_reps(path=(self.rootdir + protocol), numrep=self.nrep, which=which_fp, dbs=self.pinit)


    def likelihood_mult(self):
        """
        Multiplies likelihoods for all repetition at each fixed parameters and plots some result
        :return: saves and plots the multiplied likelihoods into disk
        """

        for protocol in self.subdirs:
            mult_likelihood(path=(self.rootdir+protocol), numfp=self.nfp, num_mult=self.nrep)
            plot_combined_results(self.rootdir+protocol, self.nfp, dbs=self.pinit)

    def combinations(self, comb_lists, output_dirs):
        """
        Multiplies likelihoods from different protocols and plots results.
        :param comb_lists: list of list of protocol names to combine \ struct: [ ["step3", "step100"], ["sin10", "sin100"] ]
        :param output_dirs: directory for the output of combinations (inside root dir) \ for examp: ["steps/comb", "sins/comb"]
        :return: Saves and plots the combinations of protocols to disk
        """

        # create lists for combine_likelihoods function
        comb_lists = [[(self.rootdir+elem) for elem in comb_lists[i]] for i in range(len(comb_lists))]
        out_dirs = [(self.rootdir+subdir) for subdir in output_dirs]

        print("Combination lists:")
        print(comb_lists)

        for idx, comblist in enumerate(comb_lists):
            combine_likelihood(comblist, numfp=self.nfp, num_mult_single=self.nrep/len(comblist), out_path=out_dirs[idx])
            plot_combined_results(out_dirs[idx], self.nfp, self.pinit)

    def compare_protocols(self, pathlist, xticks):
        """
        Compare the results of different protocols

        :param pathlist: list of protocol subdirectories to compare \ example: ["steps/3", "steps/100", "steps/comb"]
        :param xticks: list of protocol names will show on the axe of the plot \ example: ["3ms", "100ms", "scomb"]
        :return: save the plots of different protocol comparison statistics
        """
        check_directory(self.rootdir+"Results")
        pathlist = [(self.rootdir+item) for item in pathlist]
        protocol_comparison(pathlist, self.nfp, self.nrep, self.pnames, self.rootdir+"Results", self.pinit, xticks)


