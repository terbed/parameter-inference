import numpy as np
from matplotlib import pyplot as plt

cm_b = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/cm_broadness.txt")
cm_rd = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/cm_rdiff.txt")
cm_s = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/cm_sharpness.txt")

gpas_b = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/gpas_broadness.txt")
gpas_rd = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/gpas_rdiff.txt")
gpas_s = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/gpas_sharpness.txt")

Ra_b = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/Ra_broadness.txt")
Ra_rd = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/Ra_rdiff.txt")
Ra_s = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/Ra_sharpness.txt")

KL = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/KL.txt")

cm_avrg_b = np.average(cm_b[:, 1:], axis=1)
cm_std_b = np.std(cm_b[:, 1:], axis=1)
cm_avrg_s = np.average(cm_s[:, 1:], axis=1)
cm_std_s = np.std(cm_s[:, 1:], axis=1)

gpas_avrg_b = np.average(gpas_b[:, 1:], axis=1)
gpas_std_b = np.std(gpas_b[:, 1:], axis=1)
gpas_avrg_s = np.average(gpas_s[:, 1:], axis=1)
gpas_std_s = np.std(gpas_s[:, 1:], axis=1)

Ra_avrg_b = np.average(Ra_b[:, 1:], axis=1)
Ra_std_b = np.std(Ra_b[:, 1:], axis=1)
Ra_avrg_s = np.average(Ra_s[:, 1:], axis=1)
Ra_std_s = np.std(Ra_s[:, 1:], axis=1)

KL_avrg = np.average(KL[:, 1:], axis=1)
KL_std = np.std(KL[:, 1:], axis=1)

plt.figure(figsize=(12,7))
plt.title("Averaged Kullback Lieber divergence relative to the prior")
plt.xlabel("Frequency [Hz]")
plt.ylabel("KL information gain")
plt.plot(KL[:,0], KL_avrg, marker='o')
plt.errorbar(KL[:,0], KL_avrg, yerr=KL_std, linestyle='None')
plt.grid()
plt.legend(loc='best')
plt.savefig("KL_information_gain.png")

plt.figure(figsize=(12,7))
plt.title("Averaged sharpness for each paramter")
plt.xlabel("Frequency [Hz]")
plt.ylabel("sharpness")
plt.plot(cm_s[:,0], cm_avrg_s, marker='o', label = "cm", linestyle='dashed')
plt.errorbar(cm_s[:,0], cm_avrg_s, yerr=cm_std_s, linestyle='None')
plt.plot(cm_s[:,0], gpas_avrg_s, marker='v', label = "gpas", linestyle='dotted')
plt.errorbar(cm_s[:,0], gpas_avrg_s, yerr=gpas_std_s, linestyle='None')
plt.plot(cm_s[:,0], Ra_avrg_s, marker='s', label = "Ra")
plt.errorbar(cm_s[:,0], Ra_avrg_s, yerr=Ra_std_s, linestyle='None')
plt.grid()
plt.legend(loc='best')
plt.savefig("avrg_sharpness.png")

plt.figure(figsize=(12,7))
plt.title("Averaged broadness for each paramter")
plt.xlabel("Frequency [Hz]")
plt.ylabel("broadness")
plt.plot(cm_s[:,0], cm_avrg_b, marker='o', linestyle='dashed', label = "cm")
plt.errorbar(cm_s[:,0], cm_avrg_b, yerr=cm_std_b, linestyle='None')
plt.plot(cm_s[:,0], gpas_avrg_b, marker='v', linestyle='dotted', label = "gpas")
plt.errorbar(cm_s[:,0], gpas_avrg_b, yerr=gpas_std_b, linestyle='None')
plt.plot(cm_s[:,0], Ra_avrg_b, marker='s', label = "Ra")
plt.errorbar(cm_s[:,0], Ra_avrg_b, yerr=Ra_std_b, linestyle='None')
plt.grid()
plt.legend(loc='best')
plt.savefig("avrg_broadness.png")