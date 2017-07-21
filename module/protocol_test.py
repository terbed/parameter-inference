from module import plot
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood
from module.save_load import save_zipped_pickle, save_to_txt, load_zipped_pickle, load_parameter_set, save_file, \
    extend_zipped_pickle, save_params
from module.analyze import Analyse
import os
import tables as tb


def check_directory(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)


# def run_prot_sim(model, target_traces, noise_std, param_set, working_path):
#     """
#     Run multiple simulation. Use rep_traces method to generate synthetic data!
#     In this the likelihood values are evaluated for repetitions.
#
#     :param model: model function
#     :param target_traces: more_w_trace output
#     :param noise_std: Used noise standard deviation while generating sythetic data
#     :param param_set: ParameterSet object
#     :param working_path: working directory path
#     :return: Saves loglikelihood data into working directory
#     """
#
#     check_directory(working_path)
#
#     log_likelihood = []
#
#     pool = Pool(multiprocessing.cpu_count() - 1)
#     log_likelihood_func = partial(likelihood.rill, model=model, target_traces=target_traces, noise_std=noise_std)
#
#     print "Running " + str(len(param_set.parameter_set_seq)) + " simulations on all cores..."
#
#     log_likelihood = pool.map(log_likelihood_func, param_set.parameter_set_seq)
#     log_likelihood = np.array(log_likelihood)
#     pool.close()
#     pool.join()
#
#     print "log likelihood DONE!"
#
#     param_init = []
#     for param in param_set.params:
#         param_init.append(param.get_init())
#
#     data = {'params_init': param_init, 'target_traces': target_traces, 'log_likelihood': log_likelihood}
#     save_zipped_pickle(data, working_path)
#
#     print "Data SAVED!"


def run_protocol_simulations(model, target_traces, noise_std, param_set, working_path):
    """
    Run multiple simulation. Use more_w_trace method to generate synthetic data!
    In this the likelihood values are evaluated both for fixed params and repetition.

    :param model: model function
    :param target_traces: more_w_trace output
    :param noise_std: Used noise standard deviation while generating sythetic data
    :param param_set: ParameterSet obejct
    :param fixed_params: Used parameters to generate target_traces. Type: [{},{},...]
    :param working_path: working directory path
    :return: Saves loglikelihood data into working directory
    """

    check_directory(working_path)
    fpnum = target_traces.shape[0]
    rep = target_traces.shape[1]

    pool = Pool(multiprocessing.cpu_count() - 1)
    log_likelihood_func = partial(likelihood.mill, model=model, target_traces=target_traces, noise_std=noise_std)

    # Set up compressing settings
    filters = tb.Filters(complevel=6, complib='lzo')

    for idx in range(fpnum):
        save_file(target_traces[idx, :, :], working_path + "/target_traces", "tts",
                  header= "For given fixed parameter %i rep (row)" % rep)

    if param_set.isBatch is False:
        print "Running " + str(len(param_set.parameter_set_seq)) + " simulations on all cores..."

        log_likelihood = pool.map(log_likelihood_func, param_set.parameter_set_seq)
        log_likelihood = np.array(log_likelihood)
        pool.close()
        pool.join()

        print "log likelihood DONE!"

        # Save parameter initializer data
        for idx in range(fpnum):
            # Save traces log_likelihoods
            database = tb.open_file(filename=working_path + "/ll%i.hdf5" % idx, mode="w")
            database.create_carray(database.root, "ll",
                                   atom=tb.Atom.from_dtype(log_likelihood.dtype),
                                   title="loglikelihoods for %ith fixed params" % idx,
                                   shape=(log_likelihood.shape[0], log_likelihood.shape[2]), filters=filters,
                                   obj=log_likelihood[:, idx, :])
            database.flush()
            print "Data saved to disk"
            print database
            database.close()

        print "Data SAVED!"
    else:
        store = []
        lol = np.empty(shape=(2,2), dtype=np.float64)
        for idx in range(fpnum):
            # Create database for loglikelihoods
            database = tb.open_file(filename= working_path + "/ll%i.hdf5" % idx, mode="w")
            lldbs = database.create_earray(database.root, "ll",
                                           atom=tb.Atom.from_dtype(lol.dtype),
                                           title="loglikelihoods for %ith fixed params" % idx,
                                           shape=(0, rep),
                                           filters=filters,
                                           expectedrows=len(param_set.parameter_set_seq))
            store.append((lldbs, database))
        del lol

        for idx, batch in enumerate(param_set.parameter_set_batch_list):
            print str(idx) + ' batch of work is done out of ' \
                  + str(len(param_set.parameter_set_batch_list))

            batch_likelihood = pool.map(log_likelihood_func, batch)
            batch_likelihood = np.array(batch_likelihood)

            # Save batch
            for i, item in enumerate(store):
                # Save traces for fixed params
                item[0].append(batch_likelihood[:, i, :])
                item[1].flush()

        print "Data saved to disk"

        for item in store:
            item[1].close()

        pool.close()
        pool.join()
        print "log_likelihood: Done!"


def plot_single_results(path, numfp, which, dbs):
    """
    :param path: Working directory where the .hdf5 result file can be found (for given protocol)
    :param numfp: number of fixed parameters
    :param: which: a number under the number of repetition, which one will be plotted
    :param: dbs: parameter initializer hdf5 database object
    :return: Plot inference result for each parameter -- the selected one (which)
    """

    plist = []
    for idx in range(dbs.root.params_init.shape[0]):
        plist.append(dbs.root.params_init[idx, :])

    p_set = load_parameter_set(plist)

    for i in range(numfp):
        print "\n\n%i th parameter:" % i
        lldbs = tb.open_file(path + "/ll%i.hdf5", mode="r")

        for idx, param in enumerate(p_set.params):
            param.value = dbs.root.fixed_params[i, idx]

        res = Analyse(lldbs.root.ll[:, which], p_set, path+"/single_plots")
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