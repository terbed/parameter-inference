import numpy as np
from neuron import h, gui
from module.simulation import real_morphology_model
from module.probability import ParameterSet, RandomVariable
from module.noise import inv_cov_mat
from functools import partial
from module.protocol_test import run_protocol_simulations_c
import tables as tb
import time

startTime = time.time()

p_names = ['Ra', 'gpas', 'cm']
p_res = [101, 101, 101]  # Parameters resolution
p_range = [[40, 300], [0.00004, 0.00018], [0.4, 1.8]]  # Fixed range, but "true value" may change!
p_mean = [100., 0.0001, 1.]  # Fixed prior mean
p_std = [20., 0.00002, 0.2]  # Fixed prior std

noise_D = 21.6767
noise_lamb = 0.011289
dt = 0.1
samples_num = 12000


def aut_corr_func(x):
    return noise_lamb*noise_D*np.exp(-noise_lamb*np.abs(x))


noise_rep = 30       # How many repetition while params are fixed
fixed_param_num = 1  # The number of fixed parameters sampled from prior
model = real_morphology_model

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

batch_size = 30000                 # Set it to "None" to compute the whole parameter space in one stock

# Set up initial parameters (in this case only 1)
initial_params = [{'Ra': 100, 'gpas': 0.0001, 'cm': 1.}]

t_vec = np.linspace(0, (samples_num-1)*dt, samples_num)
# LOAD TARGET TRACES ...........................................
target_traces = []
for item in initial_params:
    current_param = []

    for i in range(noise_rep):
        trace = np.loadtxt("/Users/Dani/TDK/parameter_estim/param_optimizer/MULTI_TRACE/ca1pc_anat/trace%i_to_fit.txt" % i)
        current_param.append(trace[:, 1])

    target_traces.append(current_param)

target_traces = np.array(target_traces)
target_traces = target_traces.reshape((1, 30, 12000))

# LOAD STIMULUS ...............................
stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/param_optimizer/MULTI_TRACE/ca1pc_anat/stimulus_vector.txt")
print "Stimulus vector length: %i" % len(stim)

print "Constructing covmat and invcovmat..."
covmat, invcovmat = inv_cov_mat(aut_corr_func, t_vec)
print "Inverse covariance matrix is loaded to memory!"
print invcovmat.shape

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

# Save parameter informations
# Create database for data
database = tb.open_file("/Users/Dani/TDK/parameter_estim/param_optimizer/MULTI_TRACE/inference_c/paramsetup.hdf5", mode="w")

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
fixed_p = np.ndarray(shape=(len(initial_params), len(prior_set.params)))
for idx, item in enumerate(initial_params):
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


# RUN INFERENCE --------------------------------------------------------------------------------------------
working_path = "/Users/Dani/TDK/parameter_estim/param_optimizer/MULTI_TRACE/inference_c"

modell = partial(model, stim=stim)


if __name__ == '__main__':
    run_protocol_simulations_c(model=modell, target_traces=target_traces, inv_covmat=invcovmat, param_set=prior_set,
                               working_path=working_path)

runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime

