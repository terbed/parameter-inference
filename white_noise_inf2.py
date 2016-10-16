"""
- One compartment simulation
- The experimental trace is generated with white noise
- Membrane capacitance (cm) and passive conductance (gpas) is variable
- We try to infer the cm parameter and we are not interested in gpas
"""

from module.simulation import one_compartment
from module.likelihood import independent_2d as likelihood
from module.noise import white
from module.prior import normal2d
from module.trace import sharpness

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# PARAMETER SETS
cm = 1.
gpas = 0.0001

gpas_start = 0.00005
gpas_end = 0.00015
gpas_num = 80.

cm_start = 0.5
cm_end = 1.5
cm_num = 200.

gpas_mean = 0.0001
gpas_sig = 0.00002
cm_sig = 0.2
cm_mean = 1

gpas_values = np.linspace(gpas_start, gpas_end, gpas_num)
cm_values = np.linspace(cm_start, cm_end, cm_num)

gpas_step = (gpas_end - gpas_start)/gpas_num
cm_step = (cm_end - cm_start)/cm_num


# Create deterministic trace
t, v = one_compartment(cm, gpas)

# Create experimental trace with adding white noise
sigma_noise = 7
exp_v = white(sigma_noise, v)

# TRY TO INFER BACK CM PARAMETER
likelihood = likelihood(one_compartment, cm_values, gpas_values, sigma_noise, exp_v)

# Create prior distribution for cm
prior = normal2d(cm_mean, cm_sig, cm_values, gpas_mean, gpas_sig, gpas_values)

# Create posterior distribution for cm
posterior = np.multiply(likelihood, prior)
posterior = posterior / (np.sum(posterior) * cm_step * gpas_step)

# Marginalize
cm_posterior = np.sum(posterior, axis=1) * gpas_step
cm_prior = np.sum(prior, axis=1) * gpas_step

plt.figure()
plt.title("One compartment model posterior (r) and prior (b) distribution for cm ")
plt.xlabel("membrane conductance (cm) [microF/cm^2]")
plt.ylabel("probability")
plt.axvline(cm, color='g')
plt.plot(cm_values, cm_posterior, 'r')
plt.plot(cm_values, cm_prior, 'b')
plt.savefig("cm_posterior.png")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Likelihood')
plt.xlabel('gpas [mS/cm2]')
plt.ylabel('cm [microF/cm^2]')
x, y = np.meshgrid(gpas_values, cm_values)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, likelihood, rstride=1, cstride=1, cmap=CM.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("likelihood.png")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Posterior distribution')
plt.xlabel('gpas [mS/cm2]')
plt.ylabel('cm [microF/cm^2]')
x, y = np.meshgrid(gpas_values, cm_values)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, posterior, rstride=1, cstride=1, cmap=CM.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("posterior.png")

inferred_cm = cm_values[np.argmax(cm_posterior)]
posterior_sharpness = sharpness(cm_values, cm_posterior)
prior_sharpness = sharpness(cm_values, cm_prior)


print "The inferred cm value: " + str(inferred_cm)
print "The sharpness of the prior distribution: " + str(prior_sharpness)
print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
