"""
- One compartment simulation
- The experimental trace is generated with colored noise
- Membrane capacitance (cm) and passive conductance (gpas) is variable
- We try to infer the cm parameter and we are not interested in gpas
"""

from module.simulation import one_compartment
from module.likelihood import dependent_2d as likelihood
from module.noise import colored
from module.prior import normal2d
from module.trace import sharpness
import module.invcovmat as invcovmat


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm as CM


# PARAMETER SETS
cm = 1.
gpas = 0.0001

gpas_start = 0.00005
gpas_end = 0.00015
gpas_num = 200.

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

D = 30.
lamb = 0.1
dt = 0.1


# Create deterministic trace
t, v = one_compartment(cm, gpas)


# Create experimental trace with adding white noise
exp_v = colored(D, lamb, dt, v)

# Create inverse covariant matrix
print "Generating inverse covariant matrix..."
invcovmat = invcovmat.generate(D, lamb, t)

# TRY TO INFER BACK CM PARAMETER
likelihood = likelihood(one_compartment, cm_values, gpas_values, invcovmat, exp_v)

# Create prior distribution for cm
prior = normal2d(cm_mean, cm_sig, cm_values, gpas_mean, gpas_sig, gpas_values)

# Create posterior distribution for cm
posterior = np.multiply(likelihood, prior)
posterior = posterior / (np.sum(posterior) * cm_step * gpas_step)

# Marginalize
cm_posterior = np.sum(posterior, axis=1) * gpas_step
cm_prior = np.sum(prior, axis=1) * gpas_step

plt.figure()
plt.title("Deterministic (r) and noised trace (b) ")
plt.xlabel("t [ns]")
plt.ylabel("V [mV]")
plt.plot(t, v, '#A52F34')
plt.plot(t, exp_v, '#2FA5A0')
plt.savefig("/wn1/noise.png")


plt.figure()
plt.title("One compartment model posterior (r) and prior (b) distribution for cm ")
plt.xlabel("membrane conductance (cm) [microF/cm^2]")
plt.ylabel("probability")
plt.axvline(cm, color='#34A52F')
plt.plot(cm_values, cm_posterior, '#A52F34')
plt.plot(cm_values, cm_prior, color='#2FA5A0')
plt.savefig("cm_posterior.png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas_values, cm_values)
ax.plot_surface(x, y, likelihood, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, likelihood, zdir='z', offset=0, cmap=CM.coolwarm)
cset = ax.contour(x, y, likelihood, zdir='x', offset=0.00004, cmap=CM.coolwarm)
cset = ax.contour(x, y, likelihood, zdir='y', offset=1.6, cmap=CM.coolwarm)

ax.set_title('Likelihood')
ax.set_xlabel('gpas [mS/cm2]')
ax.set_ylabel('cm [microF/cm^2]')
plt.savefig("likelihood.png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas_values, cm_values)
ax.plot_surface(x, y, posterior, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, posterior, zdir='z',  offset=-0, cmap=CM.coolwarm)
cset = ax.contour(x, y, posterior, zdir='x', offset=0.00004, cmap=CM.coolwarm)
cset = ax.contour(x, y, posterior, zdir='y', offset=1.6, cmap=CM.coolwarm)

ax.set_title('Posterior')
ax.set_xlabel('gpas [mS/cm2]')
ax.set_ylabel('cm [microF/cm^2]')
plt.savefig("posterior.png")

inferred_cm = cm_values[np.argmax(cm_posterior)]
posterior_sharpness = sharpness(cm_values, cm_posterior)
prior_sharpness = sharpness(cm_values, cm_prior)


print "The inferred cm value: " + str(inferred_cm)
print "The sharpness of the prior distribution: " + str(prior_sharpness)
print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
