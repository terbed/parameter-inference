import numpy as np
from neuron import h, gui
from module.simulation import corrected_experimental_protocol_with_linear_gpas_distr
from module.probability import ParameterSet, RandomVariable
from module.noise import inv_cov_mat
from module.protocol_test import run_protocol_simulations_c
import tables as tb
import time

startTime = time.time()

p_names = ['Ra', 'gpas_soma', 'k']
p_res = [30, 30, 30]  # Parameters resolution
p_range = [[20., 500.], [0.00002, 0.0006], [[-0.0005, 0.0005]]]
p_mean = [150., 0.0003, 0.]     # Fixed prior mean
p_std = [150., 0.0003, 0.0005]  # Fixed prior std

dt = 0.1
samples_num = 15000
noise_rep = 489      # How many repetition while params are fixed
fixed_param_num = 1  # The number of fixed parameters sampled from prior

# parameters of the noise
p = [0.0947, 314.18, 822.57]


def aut_corr_func(x):
    return p[0]*np.exp(-np.abs(x)/p[1])*np.cos(2*np.pi/p[2]*np.abs(x))

# Load noise
target_traces = np.loadtxt("/home/szabolcs/parameter_inference/exp_v3/corrected_target_traces.txt")
target_traces = target_traces.T
target_traces = target_traces.reshape((1, noise_rep, samples_num))
print(target_traces.shape)


# --- Load NEURON morphology
h('load_file("/home/szabolcs/parameter_inference/exp_v3/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

batch_size = 30000                 # Set it to "None" to compute the whole parameter space in one stock

# Set up initial parameters (in this case only 1)
initial_params = [{'Ra': None, 'gpas_soma': None, 'k': None}]

t_vec = np.linspace(0, (samples_num-1)*dt, samples_num)

print("Constructing covmat and invcovmat...")
covmat, invcovmat = inv_cov_mat(aut_corr_func, t_vec)
print("Inverse covariance matrix is loaded to memory!")
print(invcovmat.shape)

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
database = tb.open_file("/home/szabolcs/parameter_inference/exp_v4/inference_c/paramsetup.hdf5", mode="w")

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
print("Parameter space initialization data saved to disk")
print(database)
database.close()


# RUN INFERENCE --------------------------------------------------------------------------------------------
working_path = "/home/szabolcs/parameter_inference/exp_v4/inference_c"


if __name__ == '__main__':
    run_protocol_simulations_c(model=corrected_experimental_protocol_with_linear_gpas_distr, target_traces=target_traces, inv_covmat=invcovmat, param_set=prior_set,
                               working_path=working_path)

runningTime = (time.time()-startTime)/60
print("\n\nThe script was running for %f minutes" % runningTime)
