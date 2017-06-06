import numpy as np
from module.plot import check_directory

hz = [1, 10, 100]
duration = [3, 20, 200]
n_rep = 10
n_par = 10

lol = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/1(0)/loglikelihood.txt")

for i in range(n_par):
    log_likelihood = np.zeros(lol.shape, dtype=np.float)
    for freq in hz:
        ll = np.loadtxt(
            "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/%i(%i)/fixed_params/loglikelihood(0).txt"
        % (freq, i))
        for j in range(n_rep-1):
            current_log_likelihood = np.loadtxt(
                "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/%i(%i)/fixed_params/loglikelihood(%i).txt"
                % (freq, i, j+1))
            ll = np.add(ll, current_log_likelihood)
        log_likelihood = np.add(log_likelihood, ll)

    check_directory("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/200(%i)" % (i))
    np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/200(%i)/loglikelihood.txt" % i,
               log_likelihood)

for i in range(n_par):
    log_likelihood = np.zeros(lol.shape, dtype=np.float)
    for dur in duration:
        ll = np.loadtxt(
            "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)/fixed_params/loglikelihood(0).txt"
        % (dur, i))
        for j in range(n_rep-1):
            current_log_likelihood = np.loadtxt(
                "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)/fixed_params/loglikelihood(%i).txt"
                % (dur, i, j+1))
            ll = np.add(ll, current_log_likelihood)
        log_likelihood = np.add(log_likelihood, ll)

    check_directory("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/300(%i)" % i)
    np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/300(%i)/loglikelihood.txt" % i,
               log_likelihood)