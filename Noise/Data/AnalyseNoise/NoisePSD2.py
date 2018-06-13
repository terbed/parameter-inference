from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

T = 0.00002
Fs = 1/T

# Set up the names of the different folders containing data
folderNames = ["15312003", "15312011", "15318006", "15320003", "15519003", "15602007", "15611018", "15618010"]
distance = ['50 um', '50 um', '65 um', '105 um', '40 um', '55 um', '135 um', '85 um']


f, axarr = plt.subplots(len(folderNames), 2, figsize=(16,20))
for idx, name in enumerate(folderNames):
    data = np.genfromtxt('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/comp_ON.txt' % name)

    f_soma, Pxx_den_soma = signal.welch(data[:, 2], Fs, window='hamming', nperseg=1000000, scaling='spectrum')
    f_dend, Pxx_den_dend = signal.welch(data[:, 1], Fs, window='hamming', nperseg=1000000, scaling='spectrum')

    # plt.figure(figsize=(12, 6))
    axarr[idx, 0].semilogy(f_soma[0:200], np.sqrt(Pxx_den_soma[0:200]), 'b', label='soma')
    axarr[idx, 0].semilogy(f_dend[0:200], np.sqrt(Pxx_den_dend[0:200]), '--g', label='dend')
    axarr[idx, 0].set_title('Noise PSD | 0-10HZ | %s %s| comp_ON' % (name, distance[idx]))
    axarr[idx, 0].set_xlabel('frequency [Hz]')
    axarr[idx, 0].set_ylabel('PSD')
    axarr[idx, 0].legend()
    axarr[idx, 0].grid()
    # plt.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/%s_comp_ON.png' % (name, name))


    data = np.genfromtxt('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/comp_OFF.txt' % name)

    f_soma, Pxx_den_soma = signal.welch(data[:, 2], Fs, window='hamming', nperseg=1000000, scaling='spectrum')
    f_dend, Pxx_den_dend = signal.welch(data[:, 1], Fs, window='hamming', nperseg=1000000, scaling='spectrum')

    # plt.figure(figsize=(12, 6))
    axarr[idx, 1].semilogy(f_soma[0:200], np.sqrt(Pxx_den_soma[0:200]), 'b', label='soma')
    axarr[idx, 1].semilogy(f_dend[0:200], np.sqrt(Pxx_den_dend[0:200]), '--g', label='dend')
    axarr[idx, 1].set_title('Noise PSD | 0-10Hz | %s %s| comp_OFF' % (name, distance[idx]))
    axarr[idx, 1].set_xlabel('frequency [Hz]')
    axarr[idx, 1].set_ylabel('PSD')
    axarr[idx, 1].legend()
    axarr[idx, 1].grid()
    # plt.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/%s/%s_comp_OFF.png' % (name, name))
f.show()
f.savefig('/home/terbe/parameter-inference/Noise/Data/AnalyseNoise/10Hz.png')
