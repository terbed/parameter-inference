"""
Inference on true experimental data
"""

from module.probability import RandomVariable, DependentInference, ParameterSet
from module.plot import plot_res as plot
import time
from neuron import h, gui
import numpy as np
from numpy import genfromtxt
from module.simulation import exp_model


# --- Set Random Variables
Ra = RandomVariable(name='Ra', range_min=90., range_max=250., resolution=40, mean=157.362128223, sigma=20, p_sampling='p')
gpas = RandomVariable(name='gpas', range_min=0.00030, range_max=0.0005, resolution=40, mean=0.000403860792541,
                      sigma=0.00002, p_sampling='p')
cm = RandomVariable(name='cm', range_min=7, range_max=8.5, resolution=40, mean=7.849480, sigma=0.2, p_sampling='p')

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = Ra.range_max
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1


# --- Load experimental trace
experimental_trace = genfromtxt("/Users/Dani/TDK/parameter_estim/exp/131117-C1_IC_1-5_subtracted_subselected)dt=0.05ms.dat",
                                skip_header=3, )
t = experimental_trace[:, 0]
exp_v = experimental_trace[:, 1]


cm_gpas = ParameterSet(cm, gpas)
Ra_gpas = ParameterSet(Ra, gpas)
Ra_cm = ParameterSet(Ra, cm)
Ra_cm_gpas = ParameterSet(Ra, cm, gpas)


inf1 = DependentInference(exp_v, cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/exp/cm-gpas_avrg")
inf2 = DependentInference(exp_v, Ra_cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-cm-gpas_avrg")
inf3 = DependentInference(exp_v, Ra_gpas, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-gpas_avrg")
inf4 = DependentInference(exp_v, Ra_cm, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-cm_avrg")


# --- Load inverse covariant matrix - [Generate inverse covariant matrix]
print "Loading inverse covariance matrix..."
startTime = time.time()
invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/invcovmat_fitted.csv')
runningTime1 = (time.time() - startTime) / 60
print "Done... Loading time was: " + str(runningTime1)

# This experimental data is averaged from N=490 independent trace, so then inv_covmat is:
N = 490  # number of experiment repetition
invcovmat = 1/N * invcovmat

startTime = time.time()
# Multiprocess simulation
if __name__ == '__main__':
    inf2.run_sim(exp_model, invcovmat)
inf2.run_evaluation()
runningTime2 = (time.time() - startTime) / 60
print "Simulation time was: " + str(runningTime2)


plot(inf2, Ra, cm)
plot(inf2, cm, gpas)
plot(inf2, Ra, gpas)
print inf2
