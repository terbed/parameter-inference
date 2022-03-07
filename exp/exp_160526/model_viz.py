import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui
from module.simulation import real_morphology_model_2
from functools import  partial
from module.probability import ParameterSet, RandomVariable

p_names = ['Ra', 'gpas', 'ffact']
p_res = [3, 3, 3]  # Parameters resolution
p_range = [[50., 200.], [0.00002, 0.0005], [1., 5.]]
p_mean = [100., 0.0001, 2.]     # Fixed prior mean
p_std = [100., 0.0003, 2.]      # Fixed prior std

dt = 0.1
samples_num = 15000
noise_rep = 45      # How many repetition while params are fixed

# create output directory
output_path = "ML_params_viz_output_2_noise"
path_target_traces = "/Users/admin/GD/PROJECTS/SPE/data/160526-C2-row-dt0.1-1.5s-corrected.txt"
path_neuron_model = 'load_file("/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526/load_new_passive.hoc")'
path_stimulus = "/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526/new_stim-dt0.1-1.5sec.txt"

# ---------------------
# Load target traces
# ---------------------
target_traces = np.loadtxt(path_target_traces)
# target_traces = target_traces.T
mean_target_trace = np.mean(target_traces, axis=0)
print(target_traces.shape, mean_target_trace.shape)

# --------------
# Load stimulus
# --------------
s = np.loadtxt(path_stimulus)
s = s[:-1]
print("stimulus len: ", s.shape)
model = partial(real_morphology_model_2, stim=s)

# -----------------------------
# --- Load NEURON morphology
# -----------------------------
h(path_neuron_model)
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

prior_params = []
for idx, item in enumerate(p_names):
    prior_params.append(RandomVariable(name=item, range_min=p_range[idx][0], range_max=p_range[idx][1],
                                       resolution=p_res[idx], sigma=p_std[idx], mean=p_mean[idx]))
paramset = ParameterSet(*prior_params)

param_combs = paramset.parameter_set_seq
print(param_combs, len(param_combs))

plt.figure(figsize=(12, 6))

model = partial(real_morphology_model_2, stim=s)
for p in param_combs:
    t, v = model(**p)
    plt.plot(t, v, "k", linewidth=1., alpha=0.5)

plt.plot(t, mean_target_trace, "r", linewidth=1.5)
plt.xlabel("Time [ms]")
plt.ylabel("Membrane potential [mV]")
plt.savefig("model_response_in_param_range.pdf")
plt.show()