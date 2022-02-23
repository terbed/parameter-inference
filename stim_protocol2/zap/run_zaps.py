import numpy as np
from module.simulation import stick_and_ball
from module.probability import RandomVariable, IndependentInference, ParameterSet
from module.noise import white
from module.plot import plot_stat, plot_joint, fullplot
from module.load import get_default_param
from functools import partial


hz = [0, 1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 150]

p_names = ['Ra', 'cm', 'gpas']
res = [40, 40, 40]
speed = 'min'
noise = 7.
n = 50
model = stick_and_ball

for item in hz:
    print("\n\n---------------------------------------- Running %i ms impulse protocol" % item)

    """
    Run statistic on given experiment protocol and parameters.
    :param model: Neuron model function
    :param stim: Stimulus for neuron
    :param noise: Noise
    :param n:  Number of iteration
    :param working_path: Working path
    :param p_names: Parameter names. For example: ['Ra', 'cm', 'gpas']
    :param res: resolution vector for params in order: [40, 40, 60]
    :param speed: 'min', 'mid' or 'max'

    :return:  Statistics
    """

    stim = np.loadtxt("/home/szabolcs/parameter_inference/stim_protocol2_v3/zap/%i/stim.txt" % item)
    working_path = "/home/szabolcs/parameter_inference/stim_protocol2_v3/zap/%i" % item

    # Do statistics for each parameter
    stat_list = []
    for _ in p_names:
        stat_list.append(np.empty((n, 6), dtype=np.float))

    # Load fixed parameters: list of parameters to be inferred
    fixed_params = []
    for name in p_names:
        fixed_params.append(get_default_param(name))

    # Inference and statistics
    for i in range(n):
        print("\n\n--------------------------------------- %i is DONE out of %i" % (i, n))

        # Set up mean and range for a cycle of simulation
        current_mean = {}
        current_minrange = []
        current_maxrange = []

        for item in fixed_params:
            if item.name == 'Ra':
                mean = np.random.normal(item.mean, item.sigma / 2.)
                current_mean['Ra'] = mean
                current_minrange.append(mean - item.offset)
                current_maxrange.append(mean + item.offset)
            else:
                mean = np.random.normal(item.mean, item.sigma)
                current_mean[item.name] = mean
                current_minrange.append(mean - item.offset)
                current_maxrange.append(mean + item.offset)

        # Biopysicsal parameters can't be negative
        for idx, item in enumerate(current_minrange):
            if item <= 0.:
                current_minrange[idx] = 0.000001

        # Set up parameters for one cycle
        current_params = []
        for idx, item in enumerate(p_names):
            current_params.append(RandomVariable(item, range_min=current_minrange[idx], range_max=current_maxrange[idx],
                                                 resolution=res[idx], mean=current_mean[item],
                                                 sigma=fixed_params[idx].sigma))

        # Generate deterministic trace and create synthetic data with noise model
        t, v = model(stype='custom', custom_stim=stim, **current_mean)
        data = white(noise, v)

        pset = ParameterSet(*current_params)
        inf = IndependentInference(data, pset, working_path=working_path, speed=speed)

        modell = partial(model, stype='custom', custom_stim=stim)
        if __name__ == '__main__':
            inf.run_sim(modell, noise)

        inf.run_evaluation()

        m = 0
        if inf.analyse_result() is None:
            print("\nCouldn't fit gauss to data!")
            for i, item in enumerate(stat_list):
                stat_list[i] = np.delete(stat_list, (i - m), axis=0)
            m += 1
        else:
            params_stat = inf.analyse_result()

            # load list of matrices with stat
            for j, stat in enumerate(params_stat):
                stat_list[j][i - m, 0], stat_list[j][i - m, 1], stat_list[j][i - m, 2], stat_list[j][i - m, 3], \
                stat_list[j][i - m, 4], stat_list[j][i - m, 5] = stat

        # Plot last signle result at the end
        if i == n - 1:
            # Marginal plots
            print(inf)

            # Fullplot
            fullplot(inf)

            # Joint plots (BRUTEFORCE for 3 parameter solution!)
            plot_joint(inf, current_params[0], current_params[1])
            plot_joint(inf, current_params[0], current_params[2])
            plot_joint(inf, current_params[1], current_params[2])

    # Plot results
    for q, item in enumerate(stat_list):
        print("Result saved to: " + working_path + "/" + p_names[q] + "_stat.txt")
        np.savetxt(working_path + "/" + p_names[q] + "_stat.txt", item,
                   header='\nsigma\tfit_err\trdiff\taccuracy\tsharper\tbroadness', delimiter='\t')

    # Save plots
    for idx, item in enumerate(stat_list):
        plot_stat(item, fixed_params[idx], working_path)
