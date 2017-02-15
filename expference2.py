"""
Inference on true experimental data
"""

from module.probability import RandomVariable, DependentInference, ParameterSet
from module.plot import plot3d as plot

from neuron import h, gui
import numpy as np
from numpy import genfromtxt
from module.simulation import exp_model


# --- Set Random Variables
Ra = RandomVariable(name='Ra', range_min=250., range_max=700., resolution=80, mean=157.362128223, sigma=40)
gpas = RandomVariable(name='gpas', range_min=0.00030, range_max=0.0006, resolution=60, mean=0.000403860792541, sigma=0.00004)
cm = RandomVariable(name='cm', range_min=5.5, range_max=9.5, resolution=50, mean=7.849480, sigma=0.4)

# --- Load NEURON morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')
# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = Ra.range_max
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1


# --- Load experimental trace
experimental_trace = genfromtxt("/Users/Dani/TDK/parameter_estim/exp/resampled_experimental_trace")
t = experimental_trace[:, 0]
exp_v = experimental_trace[:, 1]


cm_gpas = ParameterSet(cm, gpas)
Ra_cm_gpas = ParameterSet(Ra, cm, gpas)


inf1 = DependentInference(exp_v, cm_gpas)
inf2 = DependentInference(exp_v, Ra_cm_gpas)

# --- Load inverse covariant matrix - [Generate inverse covariant matrix]
print "Loading inverse covariance matrix..."
invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/inv_covmat_0.1_0.1.txt')
print "Done..."


# Multiprocess simulationm
if __name__ == '__main__':
    inf1.run_sim(exp_model, invcovmat)
inf1.run_evaluation()


# Multiprocess simulation
if __name__ == '__main__':
    inf2.run_sim(exp_model, invcovmat)
inf2.run_evaluation()


print inf1
print inf2


plot(cm, gpas, inf1.likelihood, "Likelihood (cm-gpas inference)")
plot(cm, gpas, inf1.posterior, "Posterior (cm-gpas inference)")


ra_cm_l = np.sum(inf2.likelihood, axis=2)*gpas.step
ra_cm_p = np.sum(inf2.posterior, axis=2)*gpas.step

cm_gpas_l = np.sum(inf2.likelihood, axis=0)*Ra.step
cm_gpas_p = np.sum(inf2.posterior, axis=0)*Ra.step

Ra_gpas_l = np.sum(inf2.likelihood, axis=1)*cm.step
Ra_gpas_p = np.sum(inf2.posterior, axis=1)*cm.step

plot(cm, gpas, cm_gpas_l, 'Likelihood (Ra-cm-gpas inference)', '/Users/Dani/TDK/parameter_estim/exp/out3')
plot(cm, gpas, cm_gpas_l, 'Posterior (Ra-cm-gpas inference)', '/Users/Dani/TDK/parameter_estim/exp/out3')

plot(Ra, cm, ra_cm_l, 'Likelihood (Ra-cm-gpas inference)', '/Users/Dani/TDK/parameter_estim/exp/out3')
plot(Ra, cm, ra_cm_p, 'Posterior (Ra-cm-gpas inference)')

plot(Ra, gpas, Ra_gpas_l, 'Likelihood (Ra-cm-gpas inference)', '/Users/Dani/TDK/parameter_estim/exp/out3')
plot(Ra, gpas, Ra_gpas_p, 'Posterior (Ra-cm-gpas inference)', '/Users/Dani/TDK/parameter_estim/exp/out3')
