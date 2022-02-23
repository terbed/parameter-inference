import numpy as np
from matplotlib import pyplot as plt
from neuron import h, gui
from module.simulation import real_morphology_model

# We want to answer the question: why is the sin 10Hz protocol seems to be the best
# The possible answer lies in the properties of the deterministic model changes due to input stimulus


# --- Load NEURON morphology
h('load_file("/home/terbe/parameter-inference/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 150
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1


# Create random stimulus (all frequency input is present, we are interested in the output difference)
# ----------------------------------------------------------------------------------------------------

for i in range(10):
    t = np.arange(0, 1000, 0.1)
    stim = np.random.normal(0, 0.1, len(t))

    tt, v1 = real_morphology_model(stim=stim, Ra=50, gpas=0.00005)
    _, v2 = real_morphology_model(stim=stim, Ra=100, gpas=0.0001)

    dev = np.subtract(v1, v2)

    # Watch frequency spectrum
    Ts = (t[2]-t[1])/1000               # sampling interval in msec
    Fs = 1./Ts                          # sampling frequency
    n = len(dev)                       # length of the signal in samples
    k = np.arange(n)
    L = n/Fs                            # length of signal in time [s]
    fb = 1/L                            # frequency bin
    frq = k/L                           # two sides frequency range
    frq = frq                           # one side frequency range
    frq = frq[list(range(n/2))]               # one side frequency range

    f1 = np.arange(0, 50, fb)

    Y1 = np.fft.fft(dev)/n                 # fft computing and normalization

    fig, ax = plt.subplots(4, 1, figsize=(16, 16))
    ax[0].plot(t, stim)
    ax[0].set_xlabel('Time [ms]')
    ax[0].set_ylabel('Stimulus [uA]')
    ax[0].set_title('Random white stimulus')

    ax[1].plot(tt, v1, label='Ra=50, gpas=0.00005')
    ax[1].plot(tt, v2, label='Ra=100, gpas=0.0001')
    ax[1].set_xlabel('Time [ms]')
    ax[1].set_ylabel('Stimulus [uA]')
    ax[1].set_title('Deterministic response of the model for different parameter setting')
    ax[1].legend()

    ax[2].plot(tt, dev)
    ax[2].set_xlabel('Time [ms]')
    ax[2].set_ylabel('Voltage difference [mV]')
    ax[2].set_title('Deterministic model difference')

    ax[3].plot(f1, abs(Y1[0:len(f1)]))        # plotting the spectrum
    ax[3].set_xlabel('Freq (Hz)')
    ax[3].set_ylabel('|Y(freq)|')
    ax[3].set_title('Trace in frequency domain')
    #fig.show()
    plt.savefig('modelFFT%i.png' % i)
