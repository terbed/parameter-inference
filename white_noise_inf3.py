"""
- Stick and Ball simulation
- The experimental trace is generated with white noise
- Axial Resistance (Ra) and passive conductance (gpas) is variable
- We try to infer the Ra parameter and we are not interested in gpas
"""

from module.simulation import stick_and_ball
from module.likelihood import independent_2d as likelihood
from module.noise import white
from module.prior import normal2d
from module.trace import sharpness

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
Ra_mean = 110
gpas_sig = 0.00002
gpas_mean = 0.0001

noise_sigma = 7


# Create deterministic trace
t, v = stick_and_ball(Ra, gpas, Ra_max)

# Create experimental trace with adding white noise
exp_v = white(noise_sigma, v)


# TRY TO INFER BACK Ra PARAMETER
likelihood = likelihood(stick_and_ball, Ra_values, gpas_values, noise_sigma, exp_v)

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
plt.title("Stick and ball model posterior (r) and prior (b) distribution for Ra ")
plt.xlabel("Axial resistance (Ra) [kOhm]")
plt.ylabel("probability")
plt.axvline(Ra, color='g')
plt.plot(Ra_values, Ra_posterior, 'r')
plt.plot(Ra_values, Ra_prior, 'b')
plt.savefig("/Users/Dani/TDK/parameter_estim/wn3/Ra_posterior"+str(Ra_num)+".png")

plt.figure()
plt.title("Stick and ball model likelihood (y) distribution for Ra ")
plt.xlabel("Axial resistance (Ra) [kOhm]")
plt.ylabel("probability")
plt.axvline(Ra, color='g')
plt.plot(Ra_values, Ra_likelihood, color='#A81A28')
plt.savefig("/Users/Dani/TDK/parameter_estim/wn3/Ra_likelihood"+str(Ra_num)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Likelihood')
plt.xlabel('gpas [mS/cm2]')
plt.ylabel('Ra [kOhm]')
x, y = np.meshgrid(gpas_values, Ra_values)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, likelihood, rstride=1, cstride=1, cmap=CM.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("/Users/Dani/TDK/parameter_estim/wn3/likelihood"+str(Ra_num)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Posterior distribution')
plt.xlabel('gpas [mS/cm2]')
plt.ylabel('Ra [kOhm]')
x, y = np.meshgrid(gpas_values, Ra_values)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, posterior, rstride=1, cstride=1, cmap=CM.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("/Users/Dani/TDK/parameter_estim/wn3/posterior"+str(Ra_num)+".png")

inferred_cm = Ra_values[np.argmax(Ra_posterior)]
posterior_sharpness = sharpness(Ra_values, Ra_posterior)
prior_sharpness = sharpness(Ra_values, Ra_prior)


print "The inferred Ra value: " + str(inferred_cm)
print "The sharpness of the prior distribution: " + str(prior_sharpness)
print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
