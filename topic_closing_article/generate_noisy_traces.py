from simulation import exp_model
from noise import more_c_trace
from neuron import h, gui
from matplotlib import pyplot as plt
import numpy as np

parameters = [{"Ra": 100., "gpas": 0.0001, "cm": 1.}]
noise_D = 21.6767
noise_lamb = 0.011289

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

t, _ = exp_model()


traces = more_c_trace(noise_D, noise_lamb, 0.1, exp_model, parameters, 30)

print traces.shape
traces = traces.reshape((30, 12001))
print traces.shape

plt.figure(figsize=(12, 6))
plt.plot(t, traces.transpose())

t = np.reshape(t, (len(t), 1))
print t.shape
outfile = np.concatenate((t, traces.transpose()), axis=1)
print outfile.shape

np.savetxt("traces_to_fit.txt", outfile, delimiter='\t')