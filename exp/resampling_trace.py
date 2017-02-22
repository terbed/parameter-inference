"""
Re-sampling and cutting experimental trace
dt = 0.1 is enough for inference, otherwise the data is to big to handle
"""

from matplotlib import pyplot as plt
import numpy as np
from module.trace import re_sampling


v = np.loadtxt("/Users/Dani/TDK/parameter_estim/exp/131117-C1_IC_1-5_subtracted_subselected)dt=0.05ms.dat",
               dtype=float, delimiter="\t", usecols=(0,), skiprows=3)

t = np.linspace(0, 0.05*58000, 58000)
print str(len(v))
print str(len(t))

trace = np.ndarray((len(t), 2))
trace[:, 0] = t
trace[:, 1] = v

plt.figure()
plt.title("Old high res")
plt.plot(t, v)
plt.show()


t = np.linspace(0, 1200, 12001)
new_trace = re_sampling(trace, t)


np.savetxt("/Users/Dani/TDK/parameter_estim/exp/resampled_single_trace.txt", new_trace, delimiter='\t', newline='\n')


plt.figure()
plt.title("New sampled")
plt.plot(new_trace[:, 0], new_trace[:, 1])
plt.show()
