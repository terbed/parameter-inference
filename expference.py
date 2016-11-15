"""
Inference on true experimental data
"""

import module.likelihood as l
from module.prior import normal2d
from module.trace import sharpness, interpolate
from module.noise import colored, white
import module.probability
from module.probability import RandomVariable, DependentInference, ParameterSet

from neuron import h, gui
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from module.simulation import exp_model


# --- Set Random Variables
Ra = RandomVariable(name='Ra', range_min=100., range_max=380., resolution=100, mean=157.362128223, sigma=40)
gpas = RandomVariable(name='gpas', range_min=0.00030, range_max=0.0005, resolution=100, mean=0.000403860792541, sigma=0.00004)

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

# --- Load inverse covariant matrix - [Generate inverse covariant matrix]
print "Loading inverse covariance matrix..."
invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/inv_covmat_0.1_0.1.txt')
print "Done..."

# --- Inference
paramset = ParameterSet(Ra, gpas)

Ra_gpas = DependentInference(target_trace=exp_v, parameter_set=paramset)

# Multiprocess simulation
if __name__ == '__main__':
    Ra_gpas.run_sim(exp_model, invcovmat)

Ra_gpas.run_evaluation()

# Marginalize
Ra.posterior = np.sum(Ra_gpas.posterior, axis=1) * gpas.step
gpas.posterior = np.sum(Ra_gpas.posterior, axis=0) * Ra.step
Ra_likelihood = np.sum(Ra_gpas.likelihood, axis=1) * gpas.step

plt.figure()
plt.title("Ra posterior (g), likelihood (r), prior (b)")
plt.xlabel("Axial resistance (Ra) [ohm cm]")
plt.ylabel("probability")
plt.plot(Ra.values, Ra.posterior, '#34A52F')
plt.plot(Ra.values, Ra.prior, color='#2FA5A0')
plt.plot(Ra.values, Ra_likelihood, color='#A52F34')
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/Ra_posterior_0.1c"+str(Ra.resolution)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas.values, Ra.values)
ax.plot_surface(x, y, Ra_gpas.likelihood, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, Ra_gpas.likelihood, zdir='z', offset=0, cmap=CM.coolwarm)
cset = ax.contour(x, y, Ra_gpas.likelihood, zdir='x', offset=gpas.range_min, cmap=CM.coolwarm)
cset = ax.contour(x, y, Ra_gpas.likelihood, zdir='y', offset=Ra.range_max, cmap=CM.coolwarm)
ax.set_title('Likelihood')
ax.set_xlabel('gpas [uS] ')
ax.set_ylabel('Ra [ohm cm]')
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/likelihood_0.1c"+str(Ra.resolution)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas.values, Ra.values)
ax.plot_surface(x, y, Ra_gpas.posterior, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, Ra_gpas.posterior, zdir='z',  offset=-0, cmap=CM.coolwarm)
cset = ax.contour(x, y, Ra_gpas.posterior, zdir='x', offset=gpas.range_min, cmap=CM.coolwarm)
cset = ax.contour(x, y, Ra_gpas.posterior, zdir='y', offset=Ra.range_max, cmap=CM.coolwarm)
ax.set_title('Posterior')
ax.set_xlabel('gpas [uS]')
ax.set_ylabel('Ra [ohm cm]')
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/posterior_0.1c"+str(Ra.resolution)+".png")


Ra_posterior = interpolate(Ra.values, Ra.posterior)
gpas_posterior = interpolate(gpas.values, gpas.posterior)
Ra_prior = interpolate(Ra.values, Ra.prior)
inferred_Ra = None
if type(Ra_posterior) is str:
    print "The posterior distribution is out of range in the direction: " + Ra_posterior
else:
    inferred_Ra = Ra_posterior[0][np.argmax(Ra_posterior[1])]
    posterior_sharpness = sharpness(Ra_posterior[0], Ra_posterior[1])
    prior_sharpness = sharpness(Ra_prior[0], Ra_prior[1])
    print "The inferred maximum probable Ra value: " + str(inferred_Ra)
    print "The sharpness of the prior distribution: " + str(prior_sharpness)
    print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

inferred_gpas = gpas_posterior[0, np.argmax(gpas_posterior[1, :])]
# Plot the trace of the previous and the new trace:
_, v_sim_prior = exp_model(Ra.value, gpas.value)
_, v_sim_post = exp_model(inferred_Ra, inferred_gpas)

plt.figure()
plt.title("Experimental trace (r), sim trace with prior parameters (b), sim trace with posterior parameter (g)")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [mV]")
plt.plot(experimental_trace[:, 0], experimental_trace[:, 1], 'r')
plt.plot(experimental_trace[:, 0], v_sim_prior, 'b')
plt.plot(experimental_trace[:, 0], v_sim_post, 'g')
plt.savefig('/Users/Dani/TDK/parameter_estim/exp/out/compare.png')


plt.show()
