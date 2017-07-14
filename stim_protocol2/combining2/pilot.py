import numpy as np
from module.simulation import stick_and_ball
from module.probability import ParameterSet, RandomVariable
from module.noise import more_w_trace, sampling_from_prior
from functools import partial
from module.protocol_test import run_protocol_simulations
import time

startTime = time.time()

hz = [1, 10, 100]
duration = [3, 20, 200]

p_names = ['Ra', 'cm', 'gpas']
p_res = [40, 40, 40]  # Parameters resolution
p_range = [[40, 160], [0.4, 1.6], [0.00004, 0.00016]]  # Fixed range, but "true value" may change!
p_mean = [100., 1., 0.0001]  # Fixed prior mean
p_std = [20., 0.2, 0.00002]  # Fixed prior std

noise_std = 7.
noise_rep = 30  # How many repetition while params are fixed
fixed_param_num = 10  # The number of fixed parameters sampled from prior
model = stick_and_ball

# Set up random seed
np.random.seed(42)

# Set up parameters using prior information about them (fix the range we are assuming the true parameter)
prior_params = []
for idx, item in enumerate(p_names):
    prior_params.append(RandomVariable(name=item, range_min=p_range[idx][0], range_max=p_range[idx][1],
                                       resolution=p_res[idx], sigma=p_std[idx], mean=p_mean[idx]))

prior_set = ParameterSet(*prior_params)

# Create fixed params sampled from prior
fixed_params = sampling_from_prior(prior_set, fixed_param_num)


for item in hz:
    print "\n\n---------------------------------------- Running %i Hz zap protocol" % item

    # Stimulus path
    stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/%i/stim.txt" % item)
    working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining2/zaps/%i" % item

    modell = partial(model, stype='custom', custom_stim=stim)

    # Generate synthetic data for each fixed params and given repetition
    target_traces = more_w_trace(sigma=noise_std, model=modell, params=fixed_params, rep=noise_rep)

    if __name__ == '__main__':
        run_protocol_simulations(model=modell, target_traces=target_traces, noise_std=noise_std, param_set=prior_set,
                                 fixed_params=fixed_params, working_path=working_path)

for item in duration:
    print "\n\n---------------------------------------- Running %i ms impulse protocol" % item

    # Stimulus path
    stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/%i/stim.txt" % item)
    working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining2/steps/%i" % item

    modell = partial(model, stype='custom', custom_stim=stim)

    # Generate synthetic data for each fixed params and given repetition
    target_traces = more_w_trace(sigma=noise_std, model=modell, params=fixed_params, rep=noise_rep)

    if __name__ == '__main__':
        run_protocol_simulations(model=modell, target_traces=target_traces, noise_std=noise_std, param_set=prior_set,
                                 fixed_params=fixed_params, working_path=working_path)

runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime