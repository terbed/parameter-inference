"""
Inference on true experimental data
"""

import module.likelihood as l
from module.prior import normal2d
from module.trace import sharpness, interpolate
from module.noise import colored, white
import module.probability
from module.probability import RandomVariable, DependentInference, ParameterSet

from neuron import h, gui
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from module.simulation import exp_model
from module.plot import plot3d


# --- Set Random Variables
Ra = RandomVariable(name='Ra', range_min=100., range_max=380., resolution=100, mean=157.362128223, sigma=50)
gpas = RandomVariable(name='gpas', range_min=0.00030, range_max=0.0005, resolution=100, mean=0.000403860792541, sigma=0.00005)



# --- Load experimental trace
experimental_trace = genfromtxt("/Users/Dani/TDK/parameter_estim/exp/resampled_experimental_trace")
t = experimental_trace[:, 0]
exp_v = experimental_trace[:, 1]

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = Ra.range_max
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

# --- Load inverse covariant matrix - [Generate inverse covariant matrix]
print "Loading inverse covariance matrix..."
invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/invcovmat2.txt')
print "Done..."

# --- Inference
paramset = ParameterSet(Ra, gpas)

Ra_gpas = DependentInference(target_trace=exp_v, parameter_set=paramset)

# Multiprocess simulation
if __name__ == '__main__':
    Ra_gpas.run_sim(exp_model, invcovmat)

Ra_gpas.run_evaluation()

print Ra_gpas

plot3d(Ra, gpas, Ra_gpas.likelihood, 'likelihood', "/Users/Dani/TDK/parameter_estim/exp/out2/")
plot3d(Ra, gpas, Ra_gpas.posterior, 'Posterior', "/Users/Dani/TDK/parameter_estim/exp/out2/")

