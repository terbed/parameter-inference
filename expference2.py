"""
Inference on true experimental data
"""

from module.probability import RandomVariable, DependentInference, ParameterSet, IndependentInference
from module.plot import plot_joint, fullplot
import time
from neuron import h, gui
import numpy as np
from numpy import genfromtxt
from module.simulation import exp_model

from matplotlib import pyplot as plt

# --- Set Random Variables
Ra = RandomVariable(name='Ra', range_min=100., range_max=350., resolution=60, mean=157.362128223, sigma=20)
gpas = RandomVariable(name='gpas', range_min=0.00035, range_max=0.00045, resolution=80, mean=0.000403860792541,
                      sigma=0.00002)
cm = RandomVariable(name='cm', range_min=5., range_max=10., resolution=60, mean=7.849480, sigma=0.2)

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = Ra.range_max
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1


# --- Load experimental trace
experimental_trace = genfromtxt("/Users/Dani/TDK/parameter_estim/exp/resampled_experimental_trace")   # np.ndarray()
t = experimental_trace[:, 0]
exp_v = experimental_trace[:, 1]
# #
# _, v = exp_model(cm=7., Ra=300, gpas=0.0001)
#
# plt.figure(figsize=(12,7))
# plt.title("Averaged experimental trace.")
# plt.xlabel("Time [ms]")
# plt.ylabel("Votage response [mV]")
# plt.plot(t, exp_v, color='#e45664', label="experimental data")
# plt.plot(t, v, color="#56E4D6", label="model result with parameterset:\nRa=300, cm=7, gpas=0.0001")
# plt.grid()
# plt.legend(loc="best")
# plt.savefig("/Users/Dani/TDK/parameter_estim/document2/fig/exp/exptrace.pdf")

# cm_gpas = ParameterSet(cm, gpas)
# Ra_gpas = ParameterSet(Ra, gpas)
# Ra_cm = ParameterSet(Ra, cm)
Ra_cm_gpas = ParameterSet(Ra, cm, gpas)

# inf = IndependentInference(exp_v, Ra_cm, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-cm_w2")
infw = IndependentInference(exp_model, 1., exp_v, Ra_cm_gpas,
                            working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-cm-gpas_w03", speed='min')
# inf1 = DependentInference(exp_v, cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/exp/cm-gpas")
# inf2 = DependentInference(exp_v, Ra_cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-cm-gpas")
# inf3 = DependentInference(exp_v, Ra_gpas, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-gpas")
# inf4 = DependentInference(exp_v, Ra_cm, working_path="/Users/Dani/TDK/parameter_estim/exp/Ra-cm")


# --- Load inverse covariant matrix - [Generate inverse covariant matrix]
# print "Loading inverse covariance matrix..."
# startTime = time.time()
# invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/invcovmat_fitted.csv')
# print invcovmat.shape
#
# runningTime1 = (time.time() - startTime) / 60
# print "Done... Loading time was: " + str(runningTime1)

# invcovmat = 100 * invcovmat

startTime = time.time()
# Multiprocess simulation
if __name__ == '__main__':
    infw.run_sim()

infw.run_evaluation()
print infw.analyse_result()
runningTime2 = (time.time() - startTime) / 60
print "Simulation time was: " + str(runningTime2) + " min"

fullplot(infw)
plot_joint(infw, Ra, cm)
plot_joint(infw, Ra, gpas)
plot_joint(infw, cm, gpas)
print infw
