from module import plot
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import likelihood


def protocol_test(model, target_traces, noise_std, param_set, fixed_params, working_path):
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

    log_likelihood = []

    pool = Pool(multiprocessing.cpu_count() - 1)
    log_likelihood_func = partial(likelihood.mill, model=model, target_traces=target_traces, noise_std=noise_std)
    print "Running " + str(len(param_set.parameter_set_seq)) + " simulations on all cores..."

    log_likelihood = pool.map(log_likelihood_func, param_set.parameter_set_seq)
    pool.close()
    pool.join()

    print "log likelihood DONE!"

    # Save result
    pnum = target_traces.shape[0]
    rep = target_traces.shape[1]
    log_likelihood = np.array(log_likelihood)

    for j in range(pnum):
        # Set up Fixed Params value
        for param in param_set.params:
            param.value = fixed_params[j][param.name]
        # Save parameter setups for later analysis
        plot.save_params(param_set.params, path=working_path + "/fixed_params(%i)" % j)
        for idx in range(rep):
            plot.save_file(log_likelihood[:, j, idx], working_path + "/fixed_params(%i)" % j, "loglikelihood",
                           header=str(param_set.name) + str(param_set.shape))
            plot.save_file(target_traces[j, idx, :], working_path + "/fixed_params(%i)" % j, "target_trace")

    print "Log likelihood data Saved!"
