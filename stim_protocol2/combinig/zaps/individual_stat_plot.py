import numpy as np
from matplotlib import pyplot as plt

p_num = 10

cm_b = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/cm_broadness.txt")
cm_rd = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/cm_rdiff.txt")
cm_s = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/cm_sharpness.txt")

gpas_b = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/gpas_broadness.txt")
gpas_rd = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/gpas_rdiff.txt")
gpas_s = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/gpas_sharpness.txt")

Ra_b = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/Ra_broadness.txt")
Ra_rd = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/Ra_rdiff.txt")
Ra_s = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/Ra_sharpness.txt")

plt.figure(figsize=(12, 7))
plt.title("cm Broadness (for 30 multiplied likelihood) for 10 different parameter"
          "\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Broadness")
for idx in range(p_num-1):
    plt.plot(cm_b[:,0], cm_b[:, idx+1], marker='x')
plt.savefig("cm_brod.png")

plt.figure(figsize=(12, 7))
plt.title("cm relative difference (for 30 multiplied likelihood) for 10 different parameter"
          "\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("rdiff")
for idx in range(p_num-1):
    plt.plot(cm_rd[:,0], cm_rd[:, idx+1], marker='x')
plt.savefig("cm_rdiff.png")

plt.figure(figsize=(12, 7))
plt.title("cm sharpness (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("sharpness")
for idx in range(p_num-1):
    plt.plot(cm_s[:,0], cm_s[:, idx+1], marker='x')
plt.savefig("cm_sharp.png")

plt.figure(figsize=(12, 7))
plt.title("gpas Broadness (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Broadness")
for idx in range(p_num-1):
    plt.plot(gpas_b[:,0], gpas_b[:, idx+1], marker='x')
plt.savefig("gpas_brod.png")

plt.figure(figsize=(12, 7))
plt.title("gpas relative difference (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("rdiff")
for idx in range(p_num-1):
    plt.plot(gpas_rd[:,0], gpas_rd[:, idx+1], marker='x')
plt.savefig("gpas_rdiff.png")

plt.figure(figsize=(12, 7))
plt.title("gpas sharpness (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("sharpness")
for idx in range(p_num-1):
    plt.plot(gpas_s[:,0], gpas_s[:, idx+1], marker='x')
plt.savefig("gpas_sharp.png")

plt.figure(figsize=(12, 7))
plt.title("Ra Broadness (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Broadness")
for idx in range(p_num-1):
    plt.plot(Ra_b[:,0], Ra_b[:, idx+1], marker='x')
plt.savefig("Ra_brod.png")

plt.figure(figsize=(12, 7))
plt.title("Ra relative difference (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("rdiff")
for idx in range(p_num-1):
    plt.plot(Ra_rd[:,0], Ra_rd[:, idx+1], marker='x')
plt.savefig("Ra_rdiff.png")

plt.figure(figsize=(12, 7))
plt.title("Ra sharpness (for 30 multiplied likelihood) for 10 different parameter\nand 3 different protocol (last one is combined)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("sharpness")
for idx in range(p_num-1):
    plt.plot(Ra_s[:,0], Ra_s[:, idx+1], marker='x')
plt.savefig("Ra_sharp.png")