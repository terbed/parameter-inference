import numpy as np
from probability import RandomVariable, Inference, ParameterSet, IndependentInference
from functools import partial


def get_default_param(name):
    Ra = RandomVariable(name='Ra', range_min=50., range_max=150., resolution=60, mean=100., sigma=20.)
    gpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=80, mean=0.0001, sigma=0.00002)
    cm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=60, mean=1., sigma=0.2)
    dict = {'Ra' : Ra, 'cm': cm, 'gpas': gpas}

    return dict[name]


def load_inference(loglikelihood, working_path, *param_data):
    """
    Load previous simulation results.

    :param loglikelihood: loglikelihood data of inference
    :param working_path: Where to print the results
    :param param_data: List containing: [name, range_min, range_max, resolution, mean, sigma]
    :return: inference object ready to evaluate
    """
    p = []
    for item in param_data:
        p.append(
            RandomVariable(item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])))

    pset = ParameterSet(*p)
    inf = Inference(model=None, target_trace=None, parameter_set=pset, working_path=working_path, save=False)
    inf.likelihood = loglikelihood
    print "Previous inference data result loaded!"

    return inf


def load_statistics(n, p_names, path, working_path):
    """
    Load previous simulation data and reanalyse
    :param n: number of repetition
    :param p_names: list of string for example: ['Ra', 'cm', 'gpas'] (must be the filename too: Ra(11).txt)
    :param path: path of data to be loaded
    :param working_path: path to save the results
    :return: Full statistics
    """

    # Do statistics for each parameter
    stat_list = []
    for _ in p_names:
        stat_list.append(np.empty((n, 6), dtype=np.float))

    # Load fixed parameters: list of parameters to be inferred
    fixed_params = []
    for name in p_names:
        fixed_params.append(get_default_param(name))

    # Load each inference result and calculate statistics
    for i in range(n):
        print "\n\n%i is done out of %i" % (i, n)
        # Load loglikelihood
        ll = np.loadtxt(path + "/loglikelihood(%i).txt" % i)

        # Load parameter settings
        params = []
        for name in p_names:
            params.append(np.loadtxt(path + "/%s(%i).txt" % (name, i), dtype=str))

        inf = load_inference(ll, working_path, *params)

        inf.run_evaluation()

        m = 0
        if inf.analyse_result() is None:
            print "\nCouldn't fit gauss to data!"
            for i, item in enumerate(stat_list):
                stat_list[i] = np.delete(stat_list, (i - m), axis=0)
            m += 1
        else:
            params_stat = inf.analyse_result()

            # load list of matrices with stat
            for j, stat in enumerate(params_stat):
                stat_list[j][i - m, 0], stat_list[j][i - m, 1], stat_list[j][i - m, 2], stat_list[j][i - m, 3], \
                stat_list[j][i - m, 4], stat_list[j][i - m, 5] = stat

    # Plot and save result
    for q, item in enumerate(stat_list):
        print "Result saved to: " + working_path + "/" + p_names[q] + "_stat.txt"
        np.savetxt(working_path + "/" + p_names[q] + "_stat.txt", item,
                   header='\nsigma\tfit_err\trdiff\taccuracy\tsharper\tbroadness', delimiter='\t')


if __name__ == "__main__":
    # load_statistics(50, ["Ra", "cm", "gpas"], "/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/loglikelihood",
    #                "/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp")

    cm = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/100/loglikelihood/cm(0).txt", dtype=str)
    gpas = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/100/loglikelihood/gpas(0).txt", dtype=str)
    Ra = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/100/loglikelihood/Ra(0).txt", dtype=str)
    ll = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/100/loglikelihood/loglikelihood(0).txt")

    inf = load_inference(ll, "/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/100", Ra, cm, gpas)
    inf.run_evaluation()
    print inf
    from module.plot import fullplot, plot_joint

    plot_joint(inf, inf.p.params[0], inf.p.params[1])
    plot_joint(inf, inf.p.params[0], inf.p.params[2])
    plot_joint(inf, inf.p.params[1], inf.p.params[2])
    fullplot(inf)