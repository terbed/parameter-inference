import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui
from module.simulation import real_morphology_model_2
from functools import  partial
import os

# create output directory
output_path = "ML_params_viz_output_corrected"
path_target_traces = "/Users/admin/GD/PROJECTS/SPE/data/160526-C2-row-dt0.1-1.5s-corrected.txt"
path_neuron_model = 'load_file("/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526/load_new_passive.hoc")'
path_stimulus = "/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_160526/new_stim-dt0.1-1.5sec.txt"

dt = 0.1
samples_num = 15000
noise_rep = 45      # How many repetition while params are fixed

# -----------------------------
# --- Load NEURON morphology
# -----------------------------
h(path_neuron_model)
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

# read ML params
#v p_names = ['Ra', 'gpas', 'ffact']
inferred_params = pd.read_csv("single_plots/ML.csv")
print(inferred_params.head())

# Create lis of function input dict
model_parameters = []
print
for (idx, (Ra, gpas, ffact)) in inferred_params.iterrows():
    print(idx, Ra, gpas, ffact)
    model_parameters.append({"Ra": Ra, "gpas": gpas, "ffact": ffact})

print
print(model_parameters)

# ---------------------
# Load target traces
# ---------------------
target_traces = np.loadtxt(path_target_traces)
# target_traces = target_traces.T
# target_traces = target_traces.reshape((1, noise_rep, samples_num))
print target_traces.shape


# Load stimulus
s = np.loadtxt(path_stimulus)
s = s[:-1]
print("stimulus len: ", s.shape)
model = partial(real_morphology_model_2, stim=s)


for i, trace in enumerate(target_traces):
    t, v = model(**model_parameters[i])
    plt.figure(figsize=(12, 6))
    plt.plot(t, trace, label="Recorded trace")
    plt.plot(t, v, label="Simulated trace")
    plt.legend()
    plt.title("Target trace and corresponding simulation with ML parameters | rep " + str(i))
    plt.savefig(output_path + "/rep_" + str(i) + ".pdf")

