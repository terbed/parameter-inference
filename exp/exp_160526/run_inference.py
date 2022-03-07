import numpy as np
from neuron import h, gui
from module.simulation import real_morphology_model_2
from module.probability import ParameterSet, RandomVariable
from module.noise import inv_cov_mat
from module.protocol_test import run_protocol_simulations_c
import tables as tb
from functools import  partial
import os
import time

startTime = time.time()

p_names = ['Ra', 'gpas', 'ffact']
p_res = [30, 30, 30]  # Parameters resolution
p_range = [[50., 200.], [0.00002, 0.0005], [1., 5.]]
p_mean = [100., 0.0001, 2.]     # Fixed prior mean
p_std = [100., 0.0003, 2.]      # Fixed prior std

dt = 0.1
samples_num = 15000
noise_rep = 45      # How many repetition while params are fixed
fixed_param_num = 1  # The number of fixed parameters sampled from prior
batch_size = 30000     # Set it to "None" to compute the whole parameter space in one stock

path_target_traces = "/Users/admin/GD/PROJECTS/SPE/data/160526-C2-row-dt0.1-1.5s-corrected.txt"
path_neuron_model = 'load_file("/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526/load_new_passive.hoc")'
path_stimulus = "/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526/new_stim-dt0.1-1.5sec.txt"
working_path = "/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526"

# ----------------------
# Construct noise model
# ----------------------
# def aut_corr_func(x):
#     return A*np.exp(-x/T1)*np.cos(2*np.pi/T2*x+phi)
def aut_corr_func(x):
    return A*np.exp(-np.abs(x)/T1)*np.cos(2*np.pi/T2*x)

# fitted model parameters:
A, T1, T2 = [0.023, 350, 750]

# def aut_corr_func(x):
#     return p[0]*np.exp(-np.abs(x)/p[1])*np.cos(2*np.pi/p[2]*np.abs(x))

# ---------------------
# Load target traces
# ---------------------
target_traces = np.loadtxt(path_target_traces)
# target_traces = target_traces.T
target_traces = target_traces.reshape((1, noise_rep, samples_num))
print target_traces.shape

# -----------------------------
# --- Load NEURON morphology
# -----------------------------
h(path_neuron_model)
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

# Set up initial parameters (in this case only 1)
initial_params = [{'Ra': None, 'gpas': None, 'ffact': None}]

t_vec = np.linspace(0, (samples_num-1)*dt, samples_num)

# --------------------------------------
# Construct or load invcovmat
# ---------------------------------------
invcovmat_path = None  # Set to None if covmat is to be computed
if invcovmat_path is None:
    print "Constructing covmat and invcovmat..."
    covmat, invcovmat = inv_cov_mat(aut_corr_func, t_vec)
    print "Inverse covariance matrix is loaded to memory!"
    print invcovmat.shape
    # np.savetxt("invcovmat.txt", invcovmat)
    # print("Inverse covmat is saved, next time you can load it.")
else:
    print("Loading invcovmat from given file")
    invcovmat = np.loadtxt(invcovmat_path)


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
database = tb.open_file(os.path.join(working_path, "paramsetup.hdf5"), mode="w")

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

# Load stimulus
s = np.loadtxt(path_stimulus)
s = s[:-1]
print("stimulus len: ", s.shape)
model = partial(real_morphology_model_2, stim=s)

if __name__ == '__main__':
    run_protocol_simulations_c(model=model, target_traces=target_traces, inv_covmat=invcovmat, param_set=prior_set,
                               working_path=working_path)

runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime
