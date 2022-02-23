import numpy as np
from matplotlib import pyplot as plt

Fs = 10000                             # sampling rate
Ts = 1.0/Fs                         # sampling interval
t = np.arange(0, 500, Ts)             # time vector

ff = 1                              # frequency of the signal in Hz
y = np.sin(2*np.pi*ff*t) + np.random.normal(0, 0.5, len(t))

n = len(y)                          # length of the signal in samples
k = np.arange(n)
T = n/Fs                            # length of signal in time [s]
fb = 1/T                            # frequency bin
frq = k/T                           # two sides frequency range
frq = frq[list(range(n/2))]               # one side frequency range

f = np.arange(0, 5, fb)

Y = np.fft.fft(y)/n                 # fft computing and normalization
Y = Y[list(range(n/2))]

fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].plot(t, y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(f, abs(Y[0:len(f)]))        # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
fig.show()