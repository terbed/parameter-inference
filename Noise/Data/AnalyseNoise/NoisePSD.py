from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

T = 0.00002
Fs = 1/T

# Set up the names of the different folders containing data
folderNames = ["15312003", "15312011", "15318006", "15320003", "15519003", "15602007", "15611018", "15618010"]
distance = ['50 um', '50 um', '65 um', '105 um', '40 um', '55 um', '135 um', '85 um']

for idx, name in enumerate(folderNames):
    data = np.genfromtxt('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/comp_ON.txt' % name)

    f_soma, Pxx_den_soma = signal.welch(data[:, 2], Fs, window='hamming', nperseg=1024, scaling='spectrum')
    f_dend, Pxx_den_dend = signal.welch(data[:, 1], Fs, window='hamming', nperseg=1024, scaling='spectrum')

    plt.figure(figsize=(12, 6))
    plt.semilogy(f_soma, np.sqrt(Pxx_den_soma), label='soma')
    plt.semilogy(f_dend, np.sqrt(Pxx_den_dend), label='dend')
    plt.title('Noise PSD | 0-25kH | %s %s| comp_ON' % (name, distance[idx]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid()
    plt.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/%s_comp_ON.png' % (name, name))

    f_soma, Pxx_den_soma = signal.welch(data[:, 2], Fs, window='hamming', nperseg=65536, scaling='spectrum')
    f_dend, Pxx_den_dend = signal.welch(data[:, 1], Fs, window='hamming', nperseg=65536, scaling='spectrum')

    plt.figure(figsize=(12, 6))
    plt.semilogy(f_soma[0:1500], np.sqrt(Pxx_den_soma[0:1500]), label='soma')
    plt.semilogy(f_dend[0:1500], np.sqrt(Pxx_den_dend[0:1500]), label='dend')
    plt.title('Noise PSD | 0-1kHZ | %s %s| comp_ON' % (name, distance[idx]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid()
    plt.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/%s_comp_ON_zoom.png' % (name, name))



    data = np.genfromtxt('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/comp_OFF.txt' % name)

    f_soma, Pxx_den_soma = signal.welch(data[:, 2], Fs, window='hamming', nperseg=1024, scaling='spectrum')
    f_dend, Pxx_den_dend = signal.welch(data[:, 1], Fs, window='hamming', nperseg=1024, scaling='spectrum')

    plt.figure(figsize=(12, 6))
    plt.semilogy(f_soma, np.sqrt(Pxx_den_soma), label='soma')
    plt.semilogy(f_dend, np.sqrt(Pxx_den_dend), label='dend')
    plt.title('Noise PSD | 0-25kH | %s %s| comp_OFF' % (name, distance[idx]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid()
    plt.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/%s_comp_OFF.png' % (name, name))

    f_soma, Pxx_den_soma = signal.welch(data[:, 2], Fs, window='hamming', nperseg=65536, scaling='spectrum')
    f_dend, Pxx_den_dend = signal.welch(data[:, 1], Fs, window='hamming', nperseg=65536, scaling='spectrum')

    plt.figure(figsize=(12, 6))
    plt.semilogy(f_soma[0:1500], np.sqrt(Pxx_den_soma[0:1500]), label='soma')
    plt.semilogy(f_dend[0:1500], np.sqrt(Pxx_den_dend[0:1500]), label='dend')
    plt.title('Noise PSD | 0-1kHZ | %s %s | comp_OFF' % (name, distance[idx]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid()
    plt.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/%s_comp_OFF_zoom.png' % (name, name))



