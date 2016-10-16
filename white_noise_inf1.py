from module.simulation import one_compartment
from module.likelihood import independent_1d as likelihood
from module.noise import white
from module.prior import normal
from module.trace import sharpness

import numpy as np
from matplotlib import pyplot as plt


"""
- One compartment simulation
- The experimental trace is generated with white noise
- Only membrane capacitance (cm) is variable
- We try to infer the cm value
"""

# PARAMETER SETS
cm = 1.
gpas = 0.0001

cm_start = 0.4
cm_end = 1.6
cm_num = 200

cm_mean = 1.2
cm_sig = 0.2

cm_values = np.linspace(cm_start, cm_end, cm_num)
cm_step = (cm_end - cm_start)/cm_num


# Create deterministic trace
t, v = one_compartment(cm, gpas)

# Create experimental trace with adding white noise
sigma = 7
exp_v = white(sigma, v)


# TRY TO INFER BACK CM PARAMETER
likelihood = likelihood(one_compartment, cm_values, gpas, sigma, exp_v)

# Create prior distribution for cm
prior = normal(cm_mean, cm_sig, cm_values)

# Create posterior distribution for cm
posterior = np.multiply(likelihood, prior)
posterior = posterior / (np.sum(posterior) * cm_step)


plt.figure()
plt.title("Likelihood distribution")
plt.xlabel("cm [nF]")
plt.ylabel("likelihood")
plt.axvline(cm, color='g')
plt.plot(cm_values, likelihood)
plt.savefig("likelihood.png")

plt.figure()
plt.title("Deterministic (r) and noised trace (b) ")
plt.xlabel("t [ns]")
plt.ylabel("V [mV]")
plt.plot(t, v, 'r')
plt.plot(t, exp_v, 'b.')
plt.savefig("noise.png")


plt.figure()
plt.title("One compartment model posterior (r) and prior (b) distribution for cm ")
plt.xlabel("membrane conductance (cm) [nF]")
plt.ylabel("probability")
plt.axvline(1, color='g')
plt.plot(cm_values, posterior, 'r')
plt.plot(cm_values, prior, 'b')
plt.savefig("probability.png")


inferred_cm = cm_values[np.argmax(posterior)]
posterior_sharpness = sharpness(cm_values, posterior)
prior_sharpness = sharpness(cm_values, prior)


print "The most probable cm value: " + str(inferred_cm)
print "The sharpness of the prior distribution: " + str(prior_sharpness)
print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
