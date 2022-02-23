from simulation import exp_model
from noise import more_c_trace
from neuron import h, gui
from matplotlib import pyplot as plt
import numpy as np
from sys import exit

parameters = [{"Ra": 100., "gpas": 0.0001, "cm": 1.}]
noise_D = 21.6767
noise_lamb = 0.011289
dt = 0.1

# stimulus parameters:
stim1_delay = 200
stim1_amp = 0.5
stim1_dur = 2.9

stim2_delay = 503
stim2_amp = 0.01
stim2_dur = 599.9

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 100
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

t, _ = exp_model()

# Creating stimulus vector
stim_vect = []
for time in t:
    if time >= stim1_delay and time <= stim1_delay + stim1_dur:
        stim_vect.append(stim1_amp)
    elif time >= stim2_delay and time <= stim2_delay + stim2_dur:
        stim_vect.append(stim2_amp)
    else:
        stim_vect.append(0.)

stim_vect = np.reshape(stim_vect, (len(stim_vect), 1))
np.savetxt("stimulus_vector.txt", stim_vect, delimiter='\t')

traces = more_c_trace(noise_D, noise_lamb, 0.1, exp_model, parameters, 30)

print(traces.shape)
traces = traces.reshape((30, 12001))
print(traces.shape)

plt.figure(figsize=(12, 6))
plt.plot(t, traces.transpose())

t = np.reshape(t, (len(t), 1))


for i in range(30):
    outfile = np.concatenate((t, np.reshape(traces.transpose()[:, i], t.shape)), axis=1)
    np.savetxt("trace%i_to_fit.txt" % i, outfile, delimiter='\t')

