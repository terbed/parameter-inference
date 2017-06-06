import numpy as np
from module.plot import check_directory


hz = [1, 10, 100]
duration = [3, 20, 200]
n_rep = 30
n_par = 10

for freq in hz:
    for i in range(n_par):
        log_likelihood = np.loadtxt(
            "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/%i(%i)/fixed_params/loglikelihood(0).txt"
            % (freq, i))
        for j in range(n_rep-1):
            current_log_likelihood = np.loadtxt(
                "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/%i(%i)/fixed_params/loglikelihood(%i).txt"
                % (freq, i, j+1))
            log_likelihood = np.add(log_likelihood, current_log_likelihood)

        np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/%i(%i)/loglikelihood.txt" % (freq, i),
                   log_likelihood)