import numpy as np
from matplotlib import pyplot as plt
from neuron import h, gui
from module.simulation import real_morphology_model

# We want to answer the question: why is the sin 10Hz protocol seems to be the best
# The possible answer lies in the properties of the deterministic model changes due to input stimulus

stim1 = np.loadtxt("/home/terbe/parameter-inference/stim_protocol2/stimulus/step/3/stim.txt")
stim10 = np.loadtxt("/home/terbe/parameter-inference/stim_protocol2/stimulus/sin/10/stim.txt")
stim100 = np.loadtxt("/home/terbe/parameter-inference/stim_protocol2/stimulus/sin/100/stim.txt")

tv = np.linspace(0, 500, 5001)

fig, ax = plt.subplots(3, 1, figsize=(12, 12))
ax[0].set_title("1 Hz stimulus")
ax[0].set_xlabel("Time [ms]")
ax[0].set_ylabel("I [uA]")
ax[0].plot(tv, stim1)

ax[1].set_title("10 Hz stimulus")
ax[1].set_xlabel("Time [ms]")
ax[1].set_ylabel("I [uA]")
ax[1].plot(tv, stim10)

ax[2].set_title("100 Hz stimulus")
ax[2].set_xlabel("Time [ms]")
ax[2].set_ylabel("I [uA]")
ax[2].plot(tv, stim100)
# plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/imp.png")
fig.show()

# np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/stim.txt", stim)

# print len(stim)
#

# --- Load NEURON morphology
h('load_file("/home/terbe/parameter-inference/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 150
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

t, v11 = real_morphology_model(stim=stim1, Ra=50, gpas=0.00005)
_, v12 = real_morphology_model(stim=stim1)

_, v101 = real_morphology_model(stim=stim10, Ra=50, gpas=0.00005)
_, v102 = real_morphology_model(stim=stim10)

_, v1001 = real_morphology_model(stim=stim100, Ra=50, gpas=0.00005)
_, v1002 = real_morphology_model(stim=stim100)
# v = white(7., v)


fig, ax = plt.subplots(3, 2, figsize=(16, 12))
ax[0, 0].set_title("1 Hz stimulus")
ax[0, 0].set_xlabel("Time [ms]")
ax[0, 0].set_ylabel("I [uA]")
ax[0, 0].plot(tv, stim1)

ax[0, 1].set_title("Voltage response with 2 differen parameter")
ax[0, 1].set_xlabel("Time [ms]")
ax[0, 1].set_ylabel("Voltage [mV]")
ax[0, 1].plot(t, v11, label='Ra=50, gpas=0.00005')
ax[0, 1].plot(t, v12, label='Ra=100, gpas=0.0001')
ax[0, 1].legend()

ax[1, 0].set_title("10 Hz stimulus")
ax[1, 0].set_xlabel("Time [ms]")
ax[1, 0].set_ylabel("I [uA]")
ax[1, 0].plot(tv, stim10)

ax[1, 1].set_title("Voltage response with 2 differen parameter")
ax[1, 1].set_xlabel("Time [ms]")
ax[1, 1].set_ylabel("Voltage [mV]")
ax[1, 1].plot(t, v101, label='Ra=50, gpas=0.00005')
ax[1, 1].plot(t, v102, label='Ra=100, gpas=0.0001')
ax[1, 1].legend()

ax[2, 0].set_title("100 Hz stimulus")
ax[2, 0].set_xlabel("Time [ms]")
ax[2, 0].set_ylabel("I [uA]")
ax[2, 0].plot(tv, stim100)

ax[2, 1].set_title("Voltage response with 2 differen parameter")
ax[2, 1].set_xlabel("Time [ms]")
ax[2, 1].set_ylabel("Voltage [mV]")
ax[2, 1].plot(t, v1001, label='Ra=50, gpas=0.00005')
ax[2, 1].plot(t, v1002, '--', label='Ra=100, gpas=0.0001')
ax[2, 1].legend()

# plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/imp.png")
fig.show()


dev1 = [i - j for i, j in zip(v11, v12)]
dev2 = [i - j for i, j in zip(v101, v102)]
dev3 = [i - j for i, j in zip(v1001, v1002)]

plt.figure(figsize=(12, 7))
plt.title("Voltage response difference")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [mV]")
plt.plot(t, dev1)
plt.plot(t, dev2)
plt.plot(t, dev3)
# plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/resp.png")
plt.show()


# Watch frequency spa