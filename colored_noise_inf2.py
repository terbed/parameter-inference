"""
- Stick and Ball simulation
- The experimental trace is generated with colored noise
- Axial resistance (Ra) and passive conductance (gpas) is variable
- We try to infer the Ra parameter and we are not interested in gpas
"""

from module.simulation import stick_and_ball
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


# PARAMETER SETS -----------------------
gpas = 0.0001
Ra = 100.

gpas_min = 0.00005
gpas_max = 0.00015
gpas_num = 80.

Ra_min = 50.
Ra_max = 150.
Ra_num = 100.

gpas_values = np.linspace(gpas_min, gpas_max, gpas_num)
Ra_values = np.linspace(Ra_min, Ra_max, Ra_num)

gpas_step = (gpas_max - gpas_min) / gpas_num
Ra_step = (Ra_max - Ra_min) / Ra_num

Ra_sig = 20
Ra_mean = 110
gpas_sig = 0.00002
gpas_mean = 0.0001


D = 30.
lamb = 0.1
dt = 0.1


# Create deterministic trace
t, v = stick_and_ball(Ra, gpas)


# Create experimental trace with adding colored noise
exp_v = colored(D, lamb, dt, v)

# Create inverse covariant matrix
print "Generating inverse covariant matrix..."
invcovmat = invcovmat.generate(D, lamb, t)
print "Done!"

# TRY TO INFER BACK Ra PARAMETER
likelihood = likelihood(stick_and_ball, Ra_values, gpas_values, invcovmat, exp_v)

# Create prior distribution for cm
prior = normal2d(Ra_mean, Ra_sig, Ra_values, gpas_mean, gpas_sig, gpas_values)

# Create posterior distribution for cm
posterior = np.multiply(likelihood, prior)
posterior = posterior / (np.sum(posterior) * Ra_step * gpas_step)

# Marginalize
Ra_posterior = np.sum(posterior, axis=1) * gpas_step
Ra_prior = np.sum(prior, axis=1) * gpas_step
Ra_likelihood = np.sum(likelihood, axis=1) * gpas_step

plt.figure()
plt.title("One compartment model posterior (r) and prior (b) distribution for cm ")
plt.xlabel("membrane conductance (cm) [microF/cm^2]")
plt.ylabel("probability")
plt.axvline(Ra, color='#34A52F')
plt.plot(Ra_values, Ra_posterior, '#A52F34')
plt.plot(Ra_values, Ra_prior, color='#2FA5A0')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn2/Ra_posterior"+str(Ra_num)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas_values, Ra_values)
ax.plot_surface(x, y, likelihood, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, likelihood, zdir='z', offset=0, cmap=CM.coolwarm)
cset = ax.contour(x, y, likelihood, zdir='x', offset=0.00004, cmap=CM.coolwarm)
cset = ax.contour(x, y, likelihood, zdir='y', offset=160, cmap=CM.coolwarm)


ax.set_title('Likelihood')
ax.set_xlabel('gpas [mS/cm2]')
ax.set_ylabel('Ra [kOhm]')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn2/likelihood"+str(Ra_num)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas_values, Ra_values)
ax.plot_surface(x, y, posterior, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, posterior, zdir='z',  offset=-0, cmap=CM.coolwarm)
cset = ax.contour(x, y, posterior, zdir='x', offset=0.00004, cmap=CM.coolwarm)
cset = ax.contour(x, y, posterior, zdir='y', offset=160, cmap=CM.coolwarm)

ax.set_title('Posterior')
ax.set_xlabel('gpas [mS/cm2]')
ax.set_ylabel('Ra [kOhm]')
plt.savefig("/Users/Dani/TDK/parameter_estim/cn2/posterior"+str(Ra_num)+".png")

inferred_Ra = Ra_values[np.argmax(Ra_posterior)]
posterior_sharpness = sharpness(Ra_values, Ra_posterior)
prior_sharpness = sharpness(Ra_values, Ra_prior)


print "The inferred Ra value: " + str(inferred_Ra)
print "The sharpness of the prior distribution: " + str(prior_sharpness)
print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
