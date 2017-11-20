import numpy as np
from matplotlib import pyplot
from simulation import real_morphology_model
from neuron import h
from noise import colored, white
from matplotlib import pyplot as plt


def get_step(t_vec, offset, duration):
    stim = []

    for i, t in enumerate(t_vec):
        if t < offset:
            stim.append(0.)
        elif t <= duration+offset:
            stim.append(0.01)
        else:
            stim.append(0.)

    return stim


def get_sin(t_vec, Hz):
    stim = []

    f = Hz*1e-3

    for i,t in enumerate(t_vec):
        stim.append(0.01*np.sin(2*np.pi*f*t))

    return stim

# -----------------------------------------------------------------------------------------------------------

step = [3, 20, 200]
sin = [1, 10, 100]


tv = np.linspace(0, 500, 5001)

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 160
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

for item in step:
    stim = get_step(tv, 100, item)
    # np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/step/%i/stim.txt" % item, stim)

    t, v = real_morphology_model(stim=stim)
    vc = colored(21.6767, 0.011289, 0.1, v)
    vw = white(0.48, v)

    plt.figure(figsize=(12,7))
    plt.title("Stimulus")
    plt.xlabel("Time [ms]")
    plt.ylabel("I [nA]")
    plt.plot(tv, stim)
    plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/step/%i/imp.png" % item)
    plt.show()

    plt.figure(figsize=(12,7))
    plt.title("Voltage response")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mV]")
    plt.plot(t, v)
    plt.plot(t, vw, alpha=0.5)
    plt.plot(t, vc)
    plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/step/%i/resp.png" % item)
    plt.show()

for item in sin:
    stim = get_sin(tv, item)
    np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/sin/%i/stim.txt" % item, stim)

    t, v = real_morphology_model(stim=stim)
    vc = colored(21.6767, 0.011289, 0.1, v)
    vw = white(0.5, v)

    plt.figure(figsize=(12, 7))
    plt.title("Stimulus")
    plt.xlabel("Time [ms]")
    plt.ylabel("I [nA]")
    plt.plot(tv, stim)
    plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/sin/%i/imp.png" % item)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.title("Voltage response")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mV]")
    plt.plot(t, v)
    plt.plot(t, vw, alpha=0.5)
    plt.plot(t, vc)
    plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/stimulus/sin/%i/resp.png" % item)
    plt.show()

