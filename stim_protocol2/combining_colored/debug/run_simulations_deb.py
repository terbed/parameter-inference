# I this we will test what will happen if we assume there is white noise
# This way we can check, whether the problem can be located in the colored simulations

import numpy as np
from neuron import h, gui
from module.simulation import real_morphology_model
from module.probability import ParameterSet, RandomVariable
from module.noise import more_c_trace, sampling_from_prior
from functools import partial
from module.protocol_test import run_protocol_simulations
import tables as tb
import time

startTime = time.time()

# SET UP SIMULATION PARAMETERS -------------------------------------------------------------------------------------

hz = [1, 10, 100]
duration = [3, 20, 200]

p_names = ['Ra', 'gpas']
p_res = [61, 61]  # Parameters resolution
p_range = [[40, 160], [0.00004, 0.00016]]  # Fixed range, but "true value" may change!
p_mean = [100., 0.0001]  # Fixed prior mean
p_std = [20., 0.00002]  # Fixed prior std

noise_D = 21.6767
noise_lamb = 0.011289

noise = 0.48
noise_rep = 30  # How many repetition while params are fixed
fixed_param_num = 10  # The number of fixed parameters sampled from prior
model = real_morphology_model

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = p_range[0][1]
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

batch_size = 30000                 # Set it to "None" to compute the whole parameter space in one stock

# Set up random seed
np.random.seed(42)

# invcovmat = np.genfromtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/inv_covmat.csv", delimiter=',')
# print "Inverse covariance matrix is loaded to memory!"
# print invcovmat.shape


# END OF SETTING UP SIMULATION PARAMETERS -----------------------------------------------------------------------------






# Set up parameters using prior information about them (fix the range we are assuming the true parameter)
prior_params = []
for idx, item in enumerate(p_names):
    prior_params.append(RandomVariable(name=item, range_min=p_range[idx][0], range_max=p_range[idx][1],
                                       resolution=p_res[idx], sigma=p_std[idx], mean=p_mean[idx]))

prior_set = ParameterSet(*prior_params)
prior_set.batch_len = batch_size
if batch_size != None:
    prior_set.isBatch = True
else:
    prior_set.isBatch = False
prior_set.create_batch()

# Create fixed params sampled from prior
fixed_params = sampling_from_prior(prior_set, fixed_param_num)

# Save parameter informations
# Create database for data
database = tb.open_file("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/debug/paramsetup.hdf5", mode="w")

# Save param initialization
param_init = []
for param in prior_set.params:
    param_init.append(param.get_init())
param_init = np.array(param_init, dtype=str)

database.create_array(database.root, "params_init",
                      title="Parameter space initializer. True value is about to set up!",
                      atom=tb.Atom.from_dtype(param_init.dtype),
                      shape=param_init.shape, obj=param_init)

# Save fixed params and target_traces
fixed_p = np.ndarray(shape=(len(fixed_params), len(prior_set.params)))
for idx, item in enumerate(fixed_params):
    for i, param in enumerate(prior_set.params):
        fixed_p[idx, i] = item[param.name]

database.create_array(database.root, "fixed_params",
                      title="True value for each parameter in given simulation",
                      atom=tb.Atom.from_dtype(fixed_p.dtype),
                      shape=fixed_p.shape, obj=fixed_p)

database.flush()
print "Parameter space initialization data saved to disk"
print database
database.close()


for item in hz:
    print "\n\n---------------------------------------- Running %i Hz zap protocol" % item

    # Stimulus path
    stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/sin/%i/stim.txt" % item)
    working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/debug/sins/%i" % item

    modell = partial(model, stim=stim)

    # Generate synthetic data for each fixed params and given repetition
    target_traces = more_c_trace(D=noise_D, lamb=noise_lamb, dt=0.1, model=modell, params=fixed_params, rep=noise_rep)
    print "The shape of target traces: " + str(target_traces.shape)

    if __name__ == '__main__':
        run_protocol_simulations(model=modell, target_traces=target_traces, noise_std=noise, param_set=prior_set,
                                 working_path=working_path)

for item in duration:
    print "\n\n---------------------------------------- Running %i ms impulse protocol" % item

    # Stimulus path
    stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/step/%i/stim.txt" % item)
    working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/debug/steps/%i" % item

    modell = partial(model, stim=stim)

    # Generate synthetic data for each fixed params and given repetition
    target_traces = more_c_trace(D=noise_D, lamb=noise_lamb, dt=0.1, model=modell, params=fixed_params, rep=noise_rep)

    if __name__ == '__main__':
        run_protocol_simulations(model=modell, target_traces=target_traces, noise_std=noise, param_set=prior_set,
                                 working_path=working_path)

runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime