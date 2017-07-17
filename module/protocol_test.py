from module import plot
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood
from module.save_load import save_zipped_pickle, save_to_txt, load_zipped_pickle, load_parameter_set, save_file, \
    extend_zipped_pickle
from module.analyze import Analyse
import os


def check_directory(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)


def run_protocol_simulations(model, target_traces, noise_std, param_set, fixed_params, working_path, save_txt=False):
    """
    Run multiple simulation. Use more_w_trace method to generate synthetic data!
    :param model: model function
    :param target_traces: more_w_trace output
    :param noise_std: Used noise standard deviation while generating sythetic data
    :param param_set: ParameterSet obejct
    :param fixed_params: Used parameters to generate target_traces. Type: [{},{},...]
    :param working_path: working directory path
    :return: Saves loglikelihood data into working directory
    """

    check_directory(working_path)

    log_likelihood = []

    pool = Pool(multiprocessing.cpu_count() - 1)
    log_likelihood_func = partial(likelihood.mill, model=model, target_traces=target_traces, noise_std=noise_std)

    if param_set.isBatch is False:
        print "Running " + str(len(param_set.parameter_set_seq)) + " simulations on all cores..."

        log_likelihood = pool.map(log_likelihood_func, param_set.parameter_set_seq)
        log_likelihood = np.array(log_likelihood)
        pool.close()
        pool.join()

        print "log likelihood DONE!"

        # Save result
        if save_txt:
            save_to_txt(target_traces, log_likelihood, fixed_params, param_set, working_path)

        param_init = []
        for param in param_set.params:
            param_init.append(param.get_init())

        for idx, item in enumerate(fixed_params):
            for param in param_init:  # Set up fixed param to parameter true value
                param[6] = item[param[0]]

            data = {'params_init': param_init, 'target_traces': target_traces[idx, :, :],
                    'log_likelihood': log_likelihood[:, idx, :]}
            save_zipped_pickle(data, working_path)

        print "Data SAVED!"
    else:
        for idx, batch in enumerate(param_set.parameter_set_batch_list):
            print str(idx) + ' batch of work is done out of ' \
                  + str(len(param_set.parameter_set_batch_list))

            batch_likelihood = pool.map(log_likelihood_func, batch)
            batch_likelihood = np.array(batch_likelihood)

            # Save batch
            if idx != len(param_set.parameter_set_batch_list)-1:
                for i in range(len(fixed_params)):
                    with open(working_path + "/fixed_params(%i).txt" % i, "ab") as f:
                        np.savetxt(f, batch_likelihood[:, i, :])
            else:
                for i in range(len(fixed_params)):
                    with open(working_path + "/fixed_params(%i).txt" % i, "ab") as f:
                        np.savetxt(f, batch_likelihood[:, i, :])

                param_init = []
                for param in param_set.params:
                    param_init.append(param.get_init())

                for i, item in enumerate(fixed_params):
                    for param in param_init:  # Set up fixed param to parameter true value
                        param[6] = item[param[0]]

                for i, item in enumerate(fixed_params):
                    l = np.loadtxt(working_path + "/fixed_params(%i).txt" % i)
                    os.remove(working_path + "/fixed_params(%i).txt" % i)
                    data = {'params_init': param_init, 'target_traces': target_traces[idx, :, :],
                            'log_likelihood': l}
                    save_zipped_pickle(data, working_path)

        pool.close()
        pool.join()
        print "log_likelihood: Done!"


def plot_single_results(path, numfp, which):
    """
    :param path: Working directory where the .gz result files can be found (for given protocol)
    :param numfp: number of fixed parameters (number of files in the directory)
    :param: which: a number under the number of repetition (number of loglikelihoods in the .gz files), which one will be plotted
    :return: Plot inference result for each parameter -- the selected one (which)
    """

    for i in range(numfp):
        print "\n\n%i th parameter:" % i
        data = load_zipped_pickle(path, filename="fixed_params(%i).gz" % i)
        p_set = load_parameter_set(data["params_init"])

        res = Analyse(data["log_likelihood"][:, which], p_set, path+"/single_plots")
        print res


def plot_combined_results(path, numfp):
    for i in range(numfp):
        print "\n\n%i th parameter:" % i
        data = load_zipped_pickle(path, filename="loglikelihood(%i).gz" % i)
        p_set = load_parameter_set(data["params_init"])

        res = Analyse(data["log_likelihood"], p_set, path+"/combined_plots")
        if res.get_broadness() is not None:
            print res
        else:
            print "--- Couldn't fit posterior to data!!! ---"


def mult_likelihood(path, numfp, num_mult):
    """
    Create added loglikelihood data
    :param path: Working directory where the .gz result files can be found (for given protocol)
    :param numfp: number of fixed parameters (number of files in the directory)
    :param num_mult: Number of likelihoods to be multiplied
    :return: Added loglikelihoods for each parameter and params_init pickelled (saved in path directory in .gz files)
    """

    for i in range(numfp):
        data = load_zipped_pickle(path, "fixed_params(%i).gz" % i)
        loglikelihood = data["log_likelihood"][:, 0]

        for j in range(num_mult-1):
            loglikelihood += data["log_likelihood"][:, j+1]

        d = {"params_init": data["params_init"], "log_likelihood": loglikelihood}
        save_zipped_pickle(d, path, "loglikelihood")

    print "Adding loglikelihoods DONE!"


def combine_likelihood(path_list, numfp, num_mult_single, out_path):
    """
    Create combined likelihoods
    :param path_list: Working directory where the .gz result files can be found for each protocol to combine
    :param numfp: number of fixed parameters (number of files in the directory)
    :param num_mult_single: number of likelihoods from one protocol (in path_list) to be multiplied
    :param out_path: Combined protocol likelihood .gz files will be saved in this directory
    :return: Combined protocol likelihood .gz files with params_init data in the out_path directory
    """

    for i in range(numfp):
        data = load_zipped_pickle(path_list[0], "fixed_params(%i).gz" % i)
        loglikelihood = data["log_likelihood"][:, 0]
        for idx, path in enumerate(path_list):
            if idx == 0:
                for j in range(num_mult_single - 1):
                    loglikelihood += data["log_likelihood"][:, j + 1]
            else:
                data = load_zipped_pickle(path_list[idx], "fixed_params(%i).gz" % i)
                for j in range(num_mult_single):
                    loglikelihood += data["log_likelihood"][:, j]

        d = {"params_init": data["params_init"], "log_likelihood": loglikelihood}
        save_zipped_pickle(d, out_path, "loglikelihood")

    print "Combining likelihoods DONE!"


def protocol_comparison(path_list, numfp, inferred_params, out_path):
    """
    This function compare the combined protocol results.
    :param path_list: Path for the protocols to be compared -- and where the combined loglikelihood(x).txt files can be found
    :param numfp: Number of fixed params -- the number of loglikelihood.txt files in the path 
    :param inferred_params: For example: ["Ra","cm","gpas"]
    :param out_path: The result will be saved in this directory
    :return: .txt file each contains sharpness, broadness and KL statistics for each protocol
    """
    from matplotlib import pyplot as plt
    m = 0

    KL = []  # [[],[],...] for each protocol -> [[broadness, KL],[broadness, KL],...] for each fp
    broadness = np.empty((len(path_list), numfp, len(inferred_params)))
    for idx, path in enumerate(path_list):
        tmp = []
        for j in range(numfp):
            data = load_zipped_pickle(path, "loglikelihood(%i).gz" % j)
            p_set = load_parameter_set(data["params_init"])
            res = Analyse(data["log_likelihood"], p_set, path)

            if res.get_broadness() is not None:
                for k in range(len(inferred_params)):
                    broadness[idx, j-m, k] = res.get_broadness()[k]
            else:
                m += 1

            tmp.append(res.KL)
        KL.append(tmp)

    broadness = np.array(broadness)
    KL = np.array(KL)

    # Plot KL result for each protocol
    plt.figure(figsize=(12,7))
    plt.title("Kullback Lieber Divergence test result for each protocol and fixed parameters")
    plt.xlabel("Protocol types")
    plt.ylabel("KL test")
    for i in range(numfp):
        plt.plot(range(len(path_list)), KL[:, i], marker='x')
    plt.grid()
    plt.savefig(out_path+"/KL_test.pdf")

    # Plot each fixed param result in one plot for each parameter:
    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12,7))
        plt.title(param + " results for each fixed parameter")
        plt.xlabel("Protocol types")
        plt.ylabel("Broadness")
        for i in range(numfp-m):
            plt.plot(range(len(path_list)), broadness[:, i, idx], marker='x')
        plt.grid()
        plt.savefig(out_path+"/%s_broadness.pdf" % param)

    # Create fixed parameter averaged plot
    avrg_KL = np.average(KL, axis=1)
    avrg_broad = broadness[:, 0, :]
    for i in range(numfp-m-1):
        avrg_broad += broadness[:, i+1, :]
    avrg_broad = np.divide(avrg_broad, numfp-m)

    plt.figure(figsize=(12,7))
    plt.title("Averaged Kullback Lieber Divergence test result for each protocol")
    plt.xlabel("Protocol types")
    plt.ylabel("KL test")
    plt.plot(range(len(path_list)), avrg_KL, marker='x')
    plt.grid()
    plt.savefig(out_path+"/averaged_KL_test.pdf")

    plt.figure(figsize=(12, 7))
    plt.xlabel("Protocol types")
    plt.ylabel("Broadness")
    plt.title("Averaged results for each parameter")
    for idx, param in enumerate(inferred_params):
        plt.plot(range(len(path_list)), avrg_broad[:, idx], marker='x', label=param)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(out_path+"/average_broadness.pdf")


if __name__ == "__main__":
    # plot_single_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3", 10, 7)
    #mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100", 10, 30)
    path_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/20",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/200",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/300",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/1",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/10",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100",
              "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/200"]

    # combine_likelihood(path_list, numfp=10, num_mult_single=10,
    #                    out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/200")

    #protocol_comparison(path_list, 10, ['Ra', 'cm', 'gpas'], "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3")
    plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100", 10)