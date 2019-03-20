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
from scipy.interpolate import interp1d


def check_directory(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)


def run_prot_sim(model, target_traces, noise_std, param_set, working_path):
    """
    Run multiple simulation. Use rep_traces method to generate synthetic data!
    In this the likelihood values are evaluated for repetitions.

    :param model: model function
    :param target_traces: more_w_trace output
    :param noise_std: Used noise standard deviation while generating sythetic data
    :param param_set: ParameterSet object
    :param working_path: working directory path
    :return: Saves loglikelihood data into working directory
    """

    check_directory(working_path)
    rep = target_traces.shape[0]
    filters = tb.Filters(complevel=6, complib='lzo')

    # Save zipped target traces
    save_file(target_traces, working_path + "/target_traces", "tts",
              header="For given fixed parameter %i rep (row)" % rep)

    pool = Pool(multiprocessing.cpu_count() - 1)
    log_likelihood_func = partial(likelihood.rill, model=model, target_traces=target_traces, noise_std=noise_std)

    print "Running " + str(len(param_set.parameter_set_seq)) + " simulations on all cores..."

    log_likelihood = pool.map(log_likelihood_func, param_set.parameter_set_seq)
    log_likelihood = np.array(log_likelihood)
    pool.close()
    pool.join()

    print "log likelihood DONE!"

    i = 0
    name = '{}{:d}.hdf5'.format(working_path + "/ll", i)
    while os.path.exists(name):
        i += 1
    database = tb.open_file(filename=name, mode="w")
    database.create_carray(database.root, "ll",
                           atom=tb.Atom.from_dtype(log_likelihood.dtype),
                           title="loglikelihoods for given fixed params",
                           shape=log_likelihood.shape, filters=filters,
                           obj=log_likelihood)

    print "Data SAVED!"


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
                  header="For given fixed parameter %i rep (row)" % rep)

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
        lol = np.empty(shape=(2, 2), dtype=np.float64)
        for idx in range(fpnum):
            # Create database for loglikelihoods
            database = tb.open_file(filename=working_path + "/ll%i.hdf5" % idx, mode="w")
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


def run_protocol_simulations_c(model, target_traces, inv_covmat, param_set, working_path):
    """
    Run multiple simulation. Use more_c_trace method to generate synthetic data!
    In this the likelihood values are evaluated both for fixed params and repetition.

    :param model: model function
    :param target_traces: more_w_trace output
    :param inv_covmat:
    :param param_set: ParameterSet obejct
    :param fixed_params: Used parameters to generate target_traces. Type: [{},{},...]
    :param working_path: working directory path
    :return: Saves loglikelihood data into working directory
    """

    check_directory(working_path)
    fpnum = target_traces.shape[0]
    rep = target_traces.shape[1]

    pool = Pool(multiprocessing.cpu_count() - 1)
    log_likelihood_func = partial(likelihood.mdll, model=model, target_traces=target_traces, inv_covmat=inv_covmat)

    # Set up compressing settings
    filters = tb.Filters(complevel=6, complib='lzo')

    for idx in range(fpnum):
        save_file(target_traces[idx, :, :], working_path + "/target_traces", "tts",
                  header="For given fixed parameter %i rep (row)" % rep)

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
        lol = np.empty(shape=(2, 2), dtype=np.float64)
        for idx in range(fpnum):
            # Create database for loglikelihoods
            database = tb.open_file(filename=working_path + "/ll%i.hdf5" % idx, mode="w")
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
    print "\n\n ------------------- Plot single result: %s" % path
    plist = []
    for idx in range(dbs.root.params_init.shape[0]):
        plist.append(dbs.root.params_init[idx, :])

    p_set = load_parameter_set(plist)

    for i in range(numfp):
        print "\n\n%i th parameter: --------------------------" % i
        lldbs = tb.open_file(path + "/ll%i.hdf5" % i, mode="r")

        for idx, param in enumerate(p_set.params):
            param.value = dbs.root.fixed_params[i, idx]

        res = Analyse(lldbs.root.ll[:, which], p_set, path + "/single_plots")
        print res
        lldbs.close()


def plot_combined_results(path, numfp, dbs):
    print "\n\n ------------------- Plot combined result: %s" % path
    plist = []
    for idx in range(dbs.root.params_init.shape[0]):
        plist.append(dbs.root.params_init[idx, :])

    p_set = load_parameter_set(plist)

    for i in range(numfp):
        print "\n\n%i th parameter: ---------------------" % i

        for idx, param in enumerate(p_set.params):
            param.value = dbs.root.fixed_params[i, idx]

        data = tb.open_file(path + "/llc%i.hdf5" % i, mode="r")

        res = Analyse(data.root.ll[:], p_set, path + "/combined_plots")
        data.close()

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
    print "\n\n------ Adding loglikelihoods: %s ---------" % path
    filters = tb.Filters(complevel=6, complib='lzo')
    for i in range(numfp):
        print "%i is done out of %i" % (i, numfp)
        data = tb.open_file(path + "/ll%i.hdf5" % i, mode="r")
        loglikelihood = data.root.ll[:, 0]

        for j in range(num_mult - 1):
            loglikelihood += data.root.ll[:, j + 1]

        data.close()

        store = tb.open_file(path + "/llc%i.hdf5" % i, mode="w")
        store.create_carray(store.root, "ll", atom=tb.Atom.from_dtype(loglikelihood.dtype),
                            shape=loglikelihood.shape, title="Added loglikelihoods", filters=filters, obj=loglikelihood)
        store.close()

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
    filters = tb.Filters(complevel=6, complib='lzo')
    check_directory(out_path)
    print "\n\n-------------- Combining likelihoods from more protocols -----------------"

    for i in range(numfp):
        data = tb.open_file(path_list[0] + "/ll%i.hdf5" % i, mode="r")
        loglikelihood = data.root.ll[:, 0]
        for idx, path in enumerate(path_list):
            if idx == 0:
                for j in range(num_mult_single - 1):
                    loglikelihood += data.root.ll[:, j + 1]
            else:
                data.close()
                data = tb.open_file(path_list[idx] + "/ll%i.hdf5" % i, mode="r")
                for j in range(num_mult_single):
                    loglikelihood += data.root.ll[:, j]
                data.close()

        store = tb.open_file(out_path + "/llc%i.hdf5" % i, mode="w")
        store.create_carray(store.root, name="ll", atom=tb.Atom.from_dtype(loglikelihood.dtype),
                            shape=loglikelihood.shape, title="Combined loglikelihood from more protocol",
                            filters=filters, obj=loglikelihood)
        print store
        store.close()

    print "Combining likelihoods DONE!"


def protocol_comparison(path_list, numfp, repnum_k, inferred_params, out_path, dbs,
                        protocol_xticks=['3ms', '20ms', '200ms', '1Hz', '10Hz', '100Hz', 'steps comb', 'sins comb']):
    """
    This function compare the combined protocol results.
    :param path_list: Path for the protocols to be compared -- and where the combined loglikelihood(x).txt files can be found
    :param numfp: Number of fixed params -- the number of loglikelihood.txt files in the path
    :param repnum_k: Number of repitition
    :param inferred_params: For example: ["Ra","cm","gpas"]
    :param out_path: The result will be saved in this directory
    :param dbs: paramsetup.hdf5 database object
    :param protocol_xticks: Protocol names on x axes
    :return: .txt file each contains sharpness, broadness and KL statistics for each protocol
    """
    from matplotlib import pyplot as plt
    m = 0



    plist = []
    for idx in range(dbs.root.params_init.shape[0]):
        plist.append(dbs.root.params_init[idx, :])

    p_set = load_parameter_set(plist)

    KL = []  # [[],[],...] for each protocol -> [[broadness, KL],[broadness, KL],...] for each fp

    # statistics containing  (fitted_sigma, fit_err, relative_deviation, acc, sharper, broader)
    accuracy = np.empty((len(path_list), numfp, len(inferred_params)))
    rdiff = np.empty((len(path_list), numfp, len(inferred_params)))
    broadness = np.empty((len(path_list), numfp, len(inferred_params)))
    sigma = np.empty((len(path_list), numfp, len(inferred_params)))
    repnum = np.empty((len(path_list), numfp, len(inferred_params)))

    print "\n\n broadness shape: " + str(broadness.shape)

    for idx, path in enumerate(path_list):
        tmp = []
        for j in range(numfp):
            data = tb.open_file(path + "/llc%i.hdf5" % j, mode="r")
            for l, param in enumerate(p_set.params):
                param.value = dbs.root.fixed_params[j, l]
            res = Analyse(data.root.ll, p_set, path)
            data.close()

            # for k in range(len(inferred_params)):
            broadness[idx, j, :] = res.get_broadness()
            rdiff[idx, j, :] = res.analyse_result()[:, 2]
            accuracy[idx, j, :] = res.analyse_result()[:, 3]
            sigma[idx, j, :] = res.analyse_result()[:, 0]

            tmp.append(res.KL)
        KL.append(tmp)

    # Calculate repetition number
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            for k in range(sigma.shape[2]):
                repnum[i, j, k] = est_rep_num(repnum_k, p_set.params[k].sigma, sigma[i, j, k], 10)

    broadness = np.array(broadness)
    KL = np.array(KL)
    # print("-----------------------------------------------------------------KL.shape: " + str(KL.shape))

    # Create fixed parameter averaged plot
    #avrg_KL = np.average(KL, axis=1)
    #std_KL = np.std(KL, axis=1)
    avrg_broad = np.average(broadness, axis=1)
    std_broad = np.std(broadness, axis=1)

    #avrg_acc = np.average(accuracy, axis=1)
    #std_acc = np.std(accuracy, axis=1)
    #avrg_rdiff = np.average(rdiff, axis=1)
    #std_rdiff = np.std(rdiff, axis=1)

    avrg_repnum = np.average(repnum, axis=1)
    std_repnum = np.std(repnum, axis=1)

    setup_name = path_list[0].split('/')[-3]
    setup_name = setup_name.replace('_', '-', 100)

    # repnum plot for each scenario
    for path_idx in range(avrg_repnum.shape[0]):
        splitted_path_list = path_list[path_idx].split('/')
        protocol_name = setup_name + "-" + splitted_path_list[-2] + "-" + splitted_path_list[-1]
        for param_idx in range(avrg_repnum.shape[1]):
            x = []
            y = []
            x_err = []

            d = 1
            max_rep = 1000
            prev_repnum = 0
            while prev_repnum < max_rep:
                ns = []

                for sig in sigma[path_idx, :, param_idx]:
                    n = est_rep_num(repnum_k, p_set.params[param_idx].sigma, sig, d)
                    ns.append(n)

                mean_n = np.mean(ns)
                std_n = np.std(ns)
                sn = 1/d

                x.append(int(np.round(mean_n)))
                y.append(sn)
                x_err.append(int(np.round(std_n)))

                prev_repnum = mean_n + std_n
                d += 0.1

            y = np.array(y)
            x = np.array(x)
            x_err = np.array(x_err)

            # lower upper bound
            x_min = x - x_err
            x_max = x + x_err

            f = interp1d(x, y, fill_value='extrapolate')

            # upper and lower bound traces
            y_min = f(x_min)
            y_max = f(x_max)

            plt.figure(figsize=(12, 7))
            plt.rc('text', usetex=True)
            plt.title(
                "Posterior dist. std parameter" + "  | Protocol: " + protocol_name + "  | param.: " + inferred_params[param_idx] + r" | $\sigma_{prior}$: " + str(
                    p_set.params[param_idx].sigma))
            plt.xlabel("n (number of protocol repetition)")
            plt.ylabel(r"$s_n = \sigma_{post}/\sigma_{prior}$")
            plt.plot(x, y, color="black", label=r"$s_n$ avrg")
            plt.plot(x, y_max, color="red", linestyle="--", alpha=0.5, label="error")
            plt.plot(x, y_min, color="red", linestyle="--", alpha=0.5)
            plt.axhline(y=0.1, linestyle="-.", color="blue", alpha=0.5, label=r"$s_n = \sigma_{post}/\sigma_{prior}$ = 0.1")
            plt.ylim(0, 1)
            plt.grid()
            plt.legend(loc="best")
            plt.savefig(path_list[path_idx] + "/post_std_" + inferred_params[param_idx] + ".pdf")
            plt.close()

    # for i in range(numfp - m - 1):
    #     avrg_broad += broadness[:, i + 1, :]
    # avrg_broad = np.divide(avrg_broad, numfp - m)

    # x axes for plot:
    x = range(len(path_list))
    xbox = [val+1 for val in range(len(path_list))]

    # Plot KL result for each protocol
    plt.figure(figsize=(12, 7))
    plt.title("Kullback-Leiblert Divergence test result for each protocol and fixed parameters | " + setup_name)
    plt.xlabel("Protocol types")
    plt.ylabel("KL test")
    plt.xticks(x, protocol_xticks)
    for i in range(numfp):
        plt.plot(range(len(path_list)), KL[:, i], marker='x')
    plt.grid()
    plt.savefig(out_path + "/KL_test.pdf")

    # Plot each fixed param result in one plot for each parameter:
    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12, 7))
        plt.title("Broadness for each fixed parameter | " + param + " | " + setup_name)
        plt.xlabel("Protocol types")
        plt.ylabel("Broadness")
        plt.xticks(x, protocol_xticks)
        for i in range(numfp - m):
            plt.plot(range(len(path_list)), broadness[:, i, idx], marker='x')
        plt.grid()
        plt.savefig(out_path + "/%s_broadness.pdf" % param)

    plt.figure(figsize=(12, 7))

    #plt.plot(range(len(path_list)), avrg_KL, color='b', marker='x')
    #plt.errorbar(range(len(path_list)), avrg_KL, yerr=std_KL, fmt='none', ecolor='b')
    plt.boxplot(KL.transpose())
    plt.title("Averaged Kullback-Leibler Divergence test result for each protocol | " + setup_name)
    plt.xlabel("Protocol types")
    plt.ylabel("KL test")
    plt.xticks(xbox, protocol_xticks)
    plt.grid()
    plt.savefig(out_path + "/box_KL_test.pdf")

    # Average broadness of each parameter in one plot
    plt.figure(figsize=(12, 7))
    plt.xlabel("Protocol types")
    plt.ylabel("Broadness")
    plt.xticks(x, protocol_xticks)
    plt.title("Averaged broadness for each parameter | " + setup_name)
    for idx, param in enumerate(inferred_params):
        plt.plot(range(len(path_list)), avrg_broad[:, idx], label=param, marker='x')
        plt.errorbar(range(len(path_list)), avrg_broad[:, idx], yerr=std_broad[:, idx],  fmt='none', ecolor='black')
        for i, j, k in zip(range(len(path_list)), avrg_broad[:, idx], std_broad[:, idx]):
            plt.annotate(str(int(round(j))) + "\n+/-" + str(int(round(k))), xy=(i, j), xytext=(10, 10), textcoords='offset points', color='red')
    plt.ylim(0, 100)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(out_path + "/average_broadness.pdf")

    # Box and Whisker plot of broadness for each param
    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12, 7))

        plt.boxplot(broadness[:, :, idx].transpose())
        plt.xlabel("Protocol types")
        plt.ylabel("Broadness")
        plt.xticks(xbox, protocol_xticks)
        plt.title("Posterior broadness relative to prior dist. | param.: " + param + " | " + setup_name)
        plt.ylim(0, 100)
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(out_path + "/box_broadness_%s.pdf" % param)

    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12, 7))

        #plt.plot(range(len(path_list)), avrg_acc[:, idx], label=param, color='b', marker='x')
        #plt.errorbar(range(len(path_list)), avrg_acc[:, idx], yerr=std_acc[:, idx], fmt='none', ecolor='b')
        #for i, j, k in zip(range(len(path_list)), avrg_acc[:, idx], std_acc[:, idx]):
        #    plt.annotate(str(int(round(j))) + "\n+/-" + str(int(round(k))), xy=(i, j), xytext=(10, 10), textcoords='offset points', color='red')
        plt.boxplot(accuracy[:, :, idx].transpose())
        plt.xlabel("Protocol types")
        plt.ylabel(r"Accuracy: $p_{true}/p_{infmax}*100$")
        plt.xticks(xbox, protocol_xticks)
        plt.title(("Accuracy for %s | " % param) + setup_name)
        plt.legend(loc="best")
        plt.ylim(0, 100)
        plt.grid()
        plt.savefig(out_path + "/box_accuracy_%s.pdf" % param)

    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12, 7))

        #plt.plot(range(len(path_list)), avrg_rdiff[:, idx], label=param, color='b', marker='x')
        #plt.errorbar(range(len(path_list)), avrg_rdiff[:, idx], yerr=std_rdiff[:, idx], fmt='none',  ecolor='b')
        #for i, j, k in zip(range(len(path_list)), avrg_rdiff[:, idx], std_rdiff[:, idx]):
        #    plt.annotate(str(int(round(j))) + "\n+/-" + str(int(round(k))), xy=(i, j), xytext=(10, 10), textcoords='offset points', color='red')
        plt.boxplot(rdiff[:, :, idx].transpose())
        plt.xlabel("Protocol types")
        plt.ylabel(r"Relative difference: $(inferred-true)/true*100$")
        plt.xticks(xbox, protocol_xticks)
        plt.title(("Averaged relative difference for %s | " % param) + setup_name)
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(out_path + "/box_rdiff_%s.pdf" % param)

    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12, 7))
        plt.title(r"Number of repetition needed to attain $\sigma_{post}=\sigma_{prior}/10$ | param.: " + param + " | " + setup_name)
        plt.xlabel("Protocol types")
        plt.ylabel("Repetition number")
        plt.xticks(x, protocol_xticks)
        plt.plot(range(len(path_list)), avrg_repnum[:, idx], label=(param + " prior sigma: " + str(p_set.params[idx].sigma)), color='b', marker='x')
        plt.errorbar(range(len(path_list)), avrg_repnum[:, idx], yerr=std_repnum[:, idx], fmt='none', ecolor='b')
        plt.legend(loc="best")
        for i, j, k in zip(range(len(path_list)), avrg_repnum[:, idx], std_repnum[:, idx]):
            plt.annotate(str(int(round(j))) + "\n+/-" + str(int(round(k))), xy=(i, j), xytext=(10, 10), textcoords='offset points', color='red')
        plt.yscale('log')
        plt.autoscale(True)
        plt.grid(which="both")
        plt.savefig(out_path + "/average_repnum_%s.pdf" % param)

    for idx, param in enumerate(inferred_params):
        plt.figure(figsize=(12, 7))
        bp_dict = plt.boxplot(repnum[:, :, idx].transpose())
        plt.setp(bp_dict['boxes'], color='blue')
        plt.setp(bp_dict['whiskers'], color='red')
        plt.setp(bp_dict['caps'], color='red')
        for line in bp_dict['medians']:
            # get position data for median line
            x, y = line.get_xydata()[1]
            # overlay median value
            plt.annotate(str(int(round(y))), xy=(x, y), xytext=(2, 0), textcoords='offset points', color="orangered")
        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[0]
            plt.annotate(str(int(round(y))), xy=(x, y), xytext=(-10, -10), textcoords='offset points', color="black")
            x, y = line.get_xydata()[3]
            plt.annotate(str(int(round(y))), xy=(x, y), xytext=(-10, +2), textcoords='offset points', color="black")
        plt.title(r"Number of repetition needed to achieve $\sigma_{post}=\sigma_{prior}/10$ | param.: " + param + " | " + setup_name)
        plt.xlabel("Protocol types")
        plt.ylabel("Repetition number")
        plt.yscale('log')
        plt.autoscale(True)
        plt.xticks(xbox, protocol_xticks)
        plt.grid(which="both")
        plt.savefig(out_path + "/box_repnum_%s.pdf" % param)


def est_rep_num(k, sigma_0, mean_sigma_k, d):
    """

    :param k: number of repetition through the posterior is attained
    :param sigma_0: the std parameter of the prior distribution
    :param mean_sigma_k: the mean value of the posterior distribution's std parameter
    :param d: the sigma of desired posterior distribution is sigma_0/d
    :return: the number of repetition needed for the desired posterior distribution with error
    """

    sk = mean_sigma_k/sigma_0
    sn = 1./d
    try:
        alpha = (sn**2/(1-sn**2))
    except ZeroDivisionError:
        return 0.

    n = k*(sk**2/(1-sk**2))/alpha

    return n


def inv_est_rep_num(k, sigma_0, sigma_k, n):
    """

    :param k: number of repetition through the posterior is attained
    :param sigma_0: the std parameter of the prior distribution
    :param sigma_k: std parameter assigned to kth posterior
    :param n: the repetition number is needed to achieve a posterior with given std parameter
    :return: the posteriors distribution after nth repetition divided by sigma_0
    """

    sk = sigma_k/sigma_0
    gamma = k/n*(sk**2/(1-sk**2))

    sn = np.sqrt(gamma/(1+gamma))

    return sn


def est_rep_num2(k, sk, d):
    """

    :param k: number of repetition through the posterior is attained
    :param sk: sigma_posterior_k/sigma_prior
    :param d: the sigma of desired posterior distribution is sigma_0/d
    :return: the number of repetition needed for the desired posterior distribution with error
    """

    sn = 1./d
    alpha = (sn**2/(1-sn**2))
    n = k*(sk**2/(1-sk**2))/alpha

    return n


def plot_repnum_fig(path_list, repnum, param_list):
    """
    Estimates the posterior's std in function of repetition number

    :param path_list: Path list directs to sk.csv simulation output files
    :param repnum: Number of repetition applied to given protocols
    :param param_list: list of names of inferred parameters (in order used in simulations) \ for example: ["Ra", "gpas"]
    :return: Save plots from posterior's std in function of repetition number
    """
    pass



if __name__ == "__main__":
    pinit = tb.open_file("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/paramsetup.hdf5", mode="r")
    # plot_single_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/3", 10, 7, pinit)
    # mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/200", 10, 30)
    # mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/20", 10, 30)
    # mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/3", 10, 30)
    # mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/1", 10, 30)
    # mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/10", 10, 30)
    # mult_likelihood("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/100", 10, 30)

    steps_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/3",
                  "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/20",
                  "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/200"]
    zaps_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/1",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/10",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/100"]

    # combine_likelihood(zaps_list, numfp=10, num_mult_single=10,
    #                    out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/comb")

    path_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/3",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/20",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/200",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/steps/comb",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/1",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/10",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/100",
                 "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/comb"]

    protocol_comparison(path_list, 10, ['Ra', 'cm', 'gpas'], "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4", pinit)
    pinit.close()
    # plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining4/zaps/100", 10)
