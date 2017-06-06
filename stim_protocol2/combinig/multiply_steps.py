import numpy as np
from module.plot import check_directory


hz = [1, 10, 100]
duration = [3, 20, 200]
n_rep = 30
n_par = 10

for dur in duration:
    for i in range(n_par):
        log_likelihood = np.loadtxt(
            "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)/fixed_params/loglikelihood(0).txt"
            % (dur, i))
        for j in range(n_rep - 1):
            current_log_likelihood = np.loadtxt(
                "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)/fixed_params/loglikelihood(%i).txt"
                % (dur, i, j + 1))
            log_likelihood = np.add(log_likelihood, current_log_likelihood)

        np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)/loglikelihood.txt" % (dur, i),
                   log_likelihood)