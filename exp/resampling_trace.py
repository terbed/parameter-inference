"""
Re-sampling and cutting experimental trace
dt = 0.1 is enough for inference, otherwise the data is to big to handle
"""

from matplotlib import pyplot as plt
import numpy as np
from module.trace import re_sampling

trace = np.loadtxt("/Users/Dani/TDK/parameter_estim/exp/131117-C2_short.dat",
                      dtype=float, delimiter="\t", usecols=(0, 1))


plt.figure()
plt.title("Old high res")
plt.plot(trace[:, 0], trace[:, 1])


t = np.linspace(0, 1200, 12000)
new_trace = re_sampling(trace, t)


np.savetxt("/Users/Dani/TDK/parameter_estim/exp/resampled_experimental_trace", new_trace, delimiter='\t', newline='\n')


plt.figure()
plt.title("New sampled")
plt.plot(new_trace[:, 0], new_trace[:, 1])
plt.show()
