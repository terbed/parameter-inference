"""
- 100 One compartment simulation
- The experimental trace is generated with colored noise
- Membrane capacitance (cm) and passive conductance (gpas) is variable
- We try to infer the cm parameter and we are not interested in gpas
"""

from module.simulation import one_compartment
from module.likelihood import dependent_2d as likelihood_func
from module.noise import colored
from module.prior import normal2d
from module.trace import stat

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

# PARAMETER SETS
cm = 1.
gpas = 0.0001

gpas_start = 0.00005
gpas_end = 0.00015
gpas_num = 80

cm_start = 0.5
cm_end = 1.5
cm_num = 50

gpas_mean = 0.0001
gpas_sig = 0.00002
cm_sig = 0.2
cm_mean = 1

gpas_values = np.linspace(gpas_start, gpas_end, gpas_num)
cm_values = np.linspace(cm_start, cm_end, cm_num)

gpas_step = (gpas_end - gpas_start)/gpas_num
cm_step = (cm_end - cm_start)/cm_num

D = 30.
lamb = 0.1
dt = 0.1


# Create deterministic trace
t, v = one_compartment(cm, gpas)

# Create inverse covariant matrix
print "Load inverse covariant matrix..."
invcovmat = np.genfromtxt("/Users/Dani/TDK/parameter_estim/cn1/statistics/inv_covmat0.1.csv", delimiter=',')
print "Done!"

statmat = np.zeros((3, 100), dtype=np.float)

for i in range(100):
    print i
    # Create experimental trace with adding white noise
    exp_v = colored(D, lamb, dt, v)
    # TRY TO INFER BACK CM PARAMETER
    likelihood = likelihood_func(one_compartment, cm_values, gpas_values, invcovmat, exp_v)

    # Create prior distribution for cm
    prior = normal2d(cm_mean, cm_sig, cm_values, gpas_mean, gpas_sig, gpas_values)

    # Create posterior distribution for cm
    posterior = np.multiply(likelihood, prior)
    posterior = posterior / (np.sum(posterior) * cm_step * gpas_step)

    # Marginalize
    cm_posterior = np.sum(posterior, axis=1) * gpas_step
    cm_prior = np.sum(prior, axis=1) * gpas_step

    statmat[0][i], statmat[1][i], statmat[2][i] = stat(cm_posterior, cm_prior, cm_values, cm)

sys.stdout = open('/Users/Dani/TDK/parameter_estim/cn1/statistics/statistic_result' + str(cm_num) + '.txt', 'w')
print "The distance of the most likely parameter from the true one on the average: " + str(np.average(statmat[0]))
print "The standard deviation of the upper distance: " + str(np.std(statmat[0]))
print "The inferred most probable parameter is how many times probable on the average " \
      "(according to the posterior distribution) then the true one: " + str(np.average(statmat[1]))
print "The standard deviation of the upper stat: " + str(np.std(statmat[1]))
print "The posterior distribution is how many times sharper then the prior distribution on the average: " \
      + str(np.average(statmat[2]))
print "The standard deviation of the upper sharpness: " + str(np.std(statmat[2]))

plt.figure()
plt.title("The distance of the most likely parameter from the true one")
plt.xlabel("Simulation number")
plt.ylabel("Distance")
plt.plot(range(100), statmat[0], 'bo')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn1/statistics/dist" + str(cm_num) + ".png")

plt.figure()
plt.title("The inferred most probable parameter is how many times probable")
plt.xlabel("Simulation number")
plt.ylabel("times true probability -> max inferred probability")
plt.plot(range(100), statmat[1], 'ro')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn1/statistics/p_times" + str(cm_num) + ".png")

plt.figure()
plt.title("TThe posterior distribution is how many times sharper then the prior distribution")
plt.xlabel("Simulation number")
plt.ylabel("times prior -> posterior sharpness")
plt.plot(range(100), statmat[2], 'go')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn1/statistics/sharp_times" + str(cm_num) + ".png")
