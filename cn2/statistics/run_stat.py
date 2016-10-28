"""
- 100 Stick and Ball simulation
- The experimental trace is generated with colored noise
- Axial resistance (Ra) and passive conductance (gpas) is variable
- We try to infer the Ra parameter and we are not interested in gpas
"""

from module.simulation import stick_and_ball
from module.likelihood import dependent_2d as likelihood_func
from module.noise import colored
from module.prior import normal2d
from module.trace import stat

import numpy as np
import sys
from matplotlib import pyplot as plt
import time
startTime = time.time()

# PARAMETER SETS -----------------------
gpas = 0.0001
Ra = 100.

gpas_min = 0.00005
gpas_max = 0.00015
gpas_num = 80

Ra_min = 50.
Ra_max = 150.
Ra_num = 100

gpas_values = np.linspace(gpas_min, gpas_max, gpas_num)
Ra_values = np.linspace(Ra_min, Ra_max, Ra_num)

gpas_step = (gpas_max - gpas_min) / gpas_num
Ra_step = (Ra_max - Ra_min) / Ra_num

Ra_sig = 20
Ra_mean = 100
gpas_sig = 0.00002
gpas_mean = 0.0001


D = 30.
lamb = 0.1
dt = 0.1


# Create deterministic trace
t, v = stick_and_ball(Ra, gpas)


# Create inverse covariant matrix
print "Load inverse covariant matrix..."
invcovmat = np.genfromtxt("/Users/Dani/TDK/parameter_estim/cn1/statistics/inv_covmat0.1.csv", delimiter=',')
print "Done!"

statmat = np.zeros((3, 100), dtype=np.float)

for i in range(100):
    print i
    # Create experimental trace with adding colored noise
    exp_v = colored(D, lamb, dt, v)
    # TRY TO INFER BACK Ra PARAMETER
    likelihood = likelihood_func(stick_and_ball, Ra_values, gpas_values, invcovmat, exp_v)

    # Create prior distribution for cm
    prior = normal2d(Ra_mean, Ra_sig, Ra_values, gpas_mean, gpas_sig, gpas_values)

    # Create posterior distribution for cm
    posterior = np.multiply(likelihood, prior)
    posterior = posterior / (np.sum(posterior) * Ra_step * gpas_step)

    # Marginalize
    Ra_posterior = np.sum(posterior, axis=1) * gpas_step
    Ra_prior = np.sum(prior, axis=1) * gpas_step

    statmat[0][i], statmat[1][i], statmat[2][i] = stat(Ra_posterior, Ra_prior, Ra_values, Ra)

sys.stdout = open('/Users/Dani/TDK/parameter_estim/cn2/statistics/statistic_result' + str(Ra_num) + '.txt', 'w')
print "The distance of the most likely parameter from the true one on the average: " + str(np.average(statmat[0]))
print "The standard deviation of the upper distance: " + str(np.std(statmat[0]))
print "The inferred most probable parameter is how many times probable on the average " \
      "(according to the posterior distribution) then the true one: " + str(np.average(statmat[1]))
print "The standard deviation of the upper stat: " + str(np.std(statmat[1]))
print "The posterior distribution is how many times sharper then the prior distribution on the average: " \
      + str(np.average(statmat[2]))
print "The standard deviation of the upper sharpness: " + str(np.std(statmat[2]))
runningTime = (time.time() - startTime) / 60
print "Running time: " + str(runningTime) + " min"

plt.figure()
plt.title("The distance of the most likely parameter from the true one")
plt.xlabel("Simulation number")
plt.ylabel("Distance")
plt.plot(range(100), statmat[0], 'bo')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn2/statistics/dist" + str(Ra_num) + ".png")

plt.figure()
plt.title("The inferred most probable parameter is how many times probable")
plt.xlabel("Simulation number")
plt.ylabel("times true probability -> max inferred probability")
plt.plot(range(100), statmat[1], 'ro')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn2/statistics/p_times" + str(Ra_num) + ".png")

plt.figure()
plt.title("TThe posterior distribution is how many times sharper then the prior distribution")
plt.xlabel("Simulation number")
plt.ylabel("times prior -> posterior sharpness")
plt.plot(range(100), statmat[2], 'go')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn2/statistics/sharp_times" + str(Ra_num) + ".png")
