import numpy as np
from matplotlib import pyplot as plt
import tables as tb

# 1. PART: Plot target traces
target_traces = np.genfromtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/200/target_traces/tts(0).npy.gz")

plt.figure(figsize=(12,6))
plt.plot(np.transpose(target_traces[0:10, :]))
plt.show()