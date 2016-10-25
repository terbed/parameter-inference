"""
- 100 One compartment simulation statistic
- The experimental trace is generated with white noise
- Only membrane capacitance (cm) is variable
- We try to infer the cm value
"""

from module.simulation import one_compartment
from module.likelihood import independent_1d as likelihood_func
from module.noise import white
from module.prior import normal
from module.trace import stat

import numpy as np
from matplotlib import pyplot as plt
import sys

# PARAMETER SETS
cm = 1.
gpas = 0.0001

cm_start = 0.4
cm_end = 1.6
cm_num = 100

cm_mean = 1.
cm_sig = 0.2

cm_values = np.linspace(cm_start, cm_end, cm_num)
cm_step = (cm_end - cm_start)/cm_num

# Create deterministic trace
t, v = one_compartment(cm, gpas)
sigma = 7

statmat = np.zeros((3, 100), dtype=np.float)

for i in range(100):
    print i

    # Create experimental trace with adding white noise
    exp_v = white(sigma, v)

    # TRY TO INFER BACK CM PARAMETER
    likelihood = likelihood_func(one_compartment, cm_values, gpas, sigma, exp_v)

    # Create prior distribution for cm
    prior = normal(cm_mean, cm_sig, cm_values)

    # Create posterior distribution for cm
    posterior = np.multiply(likelihood, prior)
    posterior = posterior / (np.sum(posterior) * cm_step)

    statmat[0][i], statmat[1][i], statmat[2][i] = stat(posterior, prior, cm_values, cm)

    if statmat[1][i] > 10:
        print "I am here"
        plt.figure()
        plt.title("Outstanding cases (ptimes>10)")
        plt.xlabel("Probability variable")
        plt.ylabel("Probability")
        plt.axvline(cm, color='g')
        plt.plot(cm_values, posterior)
        plt.savefig("/Users/Dani/TDK/parameter_estim/wn1/statistics/outstanding" + str(cm_num) + "_" + str(i) + ".png")


sys.stdout = open('/Users/Dani/TDK/parameter_estim/wn1/statistics/statistic_result' + str(cm_num) + '.txt', 'w')
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
plt.savefig("/Users/Dani/TDK/parameter_estim/wn1/statistics/dist" + str(cm_num) + ".png")

plt.figure()
plt.title("The inferred most probable parameter is how many times probable")
plt.xlabel("Simulation number")
plt.ylabel("times true probability -> max inferred probability")
plt.plot(range(100), statmat[1], 'ro')
plt.savefig("/Users/Dani/TDK/parameter_estim/wn1/statistics/p_times" + str(cm_num) + ".png")

plt.figure()
plt.title("TThe posterior distribution is how many times sharper then the prior distribution")
plt.xlabel("Simulation number")
plt.ylabel("times prior -> posterior sharpness")
plt.plot(range(100), statmat[2], 'go')
plt.savefig("/Users/Dani/TDK/parameter_estim/wn1/statistics/sharp_times" + str(cm_num) + ".png")
