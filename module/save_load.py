import cPickle
import gzip
import os
import numpy as np


def check_directory(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)


def save_file(X, path, name, header=''):
    check_directory(path)
    i = 0
    while os.path.exists('{}({:d}).npy.gz'.format(path + "/" + name, i)):
        i += 1
    np.savetxt('{}({:d}).npy.gz'.format(path + "/" + name, i), X, header=header, delimiter='\t')


def save_params(params, path):
    """
    :param params: List of RandomVariable objects 
    :param path: 
    :return: 
    """
    check_directory(path)
    for item in params:
        i = 0
        while os.path.exists('{}({:d}).txt'.format(path + "/" + item.name, i)):
            i += 1
        np.savetxt('{}({:d}).txt'.format(path + "/" + item.name, i), item.get_init(), fmt="%s",
                   header="name, range_min, range_max, resolution, prior_mean, prior_std, true_value")


def load_parameter_set(params_data):
    """
    Create ParameterSet object from parameter data
    :param params_data: Data list of parameters (save_params output) [param1, param2, ...] param1=param_init...
    :return: ParameterSet object 
    """
    from module.probability import RandomVariable, ParameterSet

    p = []
    for item in params_data:
        p.append(
            RandomVariable(name=item[0], range_min=float(item[1]), range_max=float(item[2]), resolution=float(item[3]),
                           mean=float(item[4]), sigma=float(item[5]), value=float(item[6])))
    p_set = ParameterSet(*p)

    return p_set


def save_to_txt(target_traces, log_likelihood, fixed_params, param_set, working_path):
    # Save result
    pnum = target_traces.shape[0]
    rep = target_traces.shape[1]
    log_likelihood = np.array(log_likelihood)

    for j in range(pnum):
        # Set up Fixed Params value
        for param in param_set.params:
            param.value = fixed_params[j][param.name]

        # Save parameter setups for later analysis
        save_params(param_set.params, path=working_path + "/fixed_params(%i)" % j)
        for idx in range(rep):
            save_file(log_likelihood[:, j, idx], working_path + "/fixed_params(%i)" % j, "loglikelihood",
                      header=str(param_set.name) + str(param_set.shape))
            save_file(target_traces[j, idx, :], working_path + "/fixed_params(%i)" % j, "target_trace")

    print "Log likelihood data Saved!"


def save_zipped_pickle(obj, path, filename="fixed_params", protocol=-1):
    check_directory(path)
    i = 0
    while os.path.exists('{}({:d}).gz'.format(path + '/' + filename, i)):
        i += 1
    with gzip.open('{}({:d}).gz'.format(path + '/' + filename, i), 'wb') as f:
        cPickle.dump(obj, f, protocol)


def extend_zipped_pickle(obj, path, filename, protocol=-1):
    check_directory(path)
    with gzip.open('{}.gz'.format(path + '/' + filename), 'wb') as f:
        cPickle.dump(obj, f, protocol)


def unzip(path, filename):
    d = filename.split('.')
    check_directory(path + "/" + d[0])
    data = load_zipped_pickle(path, filename)

    for item in data["params_init"]:
        np.savetxt('{}.txt'.format(path + "/" + d[0] + "/" + item[0]), item, fmt="%s",
                   header="name, range_min, range_max, resolution, prior_mean, prior_std, true_value")

    rep = data["target_traces"].shape[0]
    for idx in range(rep):
        save_file(data["log_likelihood"][:, idx], path + "/" + d[0], "loglikelihood")
        save_file(data["target_traces"][idx, :], path + "/" + d[0], "target_trace")


def load_zipped_pickle(path, filename):
    with gzip.open(path+'/'+filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

#if __name__ == "__main__":
