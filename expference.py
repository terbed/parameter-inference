"""
Inference on true experimental data
"""

import module.likelihood as l
from module.prior import normal2d
from module.trace import sharpness, interpolate
from module.noise import colored, white
from module.probability import RandomVariable

from neuron import h, gui
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import multiprocessing
from multiprocessing import Pool
from functools import partial

def experiment_sim(Ra, gpas, dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = 7.84948013251   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    h.psection()

    # Stimulus
    stim1 = h.IClamp(h.soma(0.01))
    stim1.delay = 200
    stim1.amp = 0.5
    stim1.dur = 2.9

    stim2 = h.IClamp(h.soma(0.01))
    stim2.delay = 503
    stim2.amp = 0.01
    stim2.dur = 599.9

    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.tstop = 1200  # Simulation end
    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


# GLOBAL PARAMETERS


# Load morphology
h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')

# Set the appropriate "nseg"
for sec in h.allsec():
    sec.Ra = 200
h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

# Colored noise parameters
#D = 0.0001
#lamb = 0.1
#dt = 0.1


noise_sigma = 1

Ra = RandomVariable(range_min=40., range_max=260., resolution=50, mean=157.362128223, sigma=30, is_target=True)
gpas = RandomVariable(range_min=0.00030, range_max=0.0005, resolution=50, mean=0.000403860792541, sigma=0.00003)

# Load experimental trace
experimental_trace = genfromtxt("/Users/Dani/TDK/parameter_estim/exp/resampled_experimental_trace")
plt.figure()
plt.title("Experimental trace")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [mV]")
plt.plot(experimental_trace[:, 0], experimental_trace[:, 1])
#plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/experimental_trace.png")


# Check simulation trace
t_sim, v_sim = experiment_sim(Ra.value, gpas.value)
plt.figure()
plt.title("Simulation trace")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [mV]")
plt.plot(t_sim, v_sim)
#plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/simulated_trace" + str(Ra.value) + "_gpas" + str(gpas.value) + ".png")
plt.show()


t = experimental_trace[:, 0]
exp_v = experimental_trace[:, 1]

run = False
if run:
    for i, x in enumerate(Ra.values):
        for j, y in enumerate(gpas.values):
            if i%10 == 0 and j%10 == 0:
                _, v_sim = experiment_sim(x, y)
                plt.figure()
                plt.plot(t, exp_v, 'r')
                plt.plot(t, v_sim, 'g')
                plt.savefig('/Users/Dani/TDK/parameter_estim/exp/out/trace_Ra' + str(x) + "_gpas" + str(y) + ".png")


# Load inverse covariant matrix - [Generate inverse covariant matrix]
# invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/inv_covmat_0.0001_0.1.txt')

shape = (len(Ra.values), len(gpas.values))


def work(param_set, simulation_func, target_trace):
    (_, v) = simulation_func(param_set[0], param_set[1])
    v_dev = np.subtract(target_trace, v)
    return np.exp(- np.sum(np.square(v_dev)) / (2 * noise_sigma ** 2))


param_seq = []
for x in Ra.values:
    for y in gpas.values:
        param_seq.append((x, y))

likelihood = []
# Multi processing
if __name__ == '__main__':
    pool = Pool(multiprocessing.cpu_count())
    likelihood = pool.map(partial(work, simulation_func=experiment_sim, target_trace=exp_v), param_seq)
    pool.close()
    pool.join()

print type(likelihood)
print likelihood
likelihood = np.reshape(likelihood, shape)

# Create prior distribution for cm
prior = normal2d(Ra.mean, Ra.sigma, Ra.values, gpas.mean, gpas.sigma, gpas.values)

# Create posterior distribution for cm
posterior = np.multiply(likelihood, prior)
posterior = posterior / (np.sum(posterior) * Ra.step * gpas.step)

# Marginalize
Ra.posterior = np.sum(posterior, axis=1) * gpas.step
Ra_likelihood = np.sum(likelihood, axis=1) * gpas.step

plt.figure()
plt.title("Ra posterior (g), likelihood (r), prior (b)")
plt.xlabel("Axial resistance (Ra) [ohm cm]")
plt.ylabel("probability")
plt.plot(Ra.values, Ra.posterior, '#34A52F')
plt.plot(Ra.values, Ra.prior, color='#2FA5A0')
plt.plot(Ra.values, Ra_likelihood, color='#A52F34')
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/Ra_posterior"+str(Ra.resolution)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas.values, Ra.values)
ax.plot_surface(x, y, likelihood, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, likelihood, zdir='z', offset=0, cmap=CM.coolwarm)
cset = ax.contour(x, y, likelihood, zdir='x', offset=0.00004, cmap=CM.coolwarm)
cset = ax.contour(x, y, likelihood, zdir='y', offset=160, cmap=CM.coolwarm)
ax.set_title('Likelihood')
ax.set_xlabel('gpas [uS] ')
ax.set_ylabel('Ra [ohm cm]')
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/likelihood"+str(Ra.resolution)+".png")


fig = plt.figure()
ax = fig.gca(projection='3d')
x, y = np.meshgrid(gpas.values, Ra.values)
ax.plot_surface(x, y, posterior, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, posterior, zdir='z',  offset=-0, cmap=CM.coolwarm)
cset = ax.contour(x, y, posterior, zdir='x', offset=0.00004, cmap=CM.coolwarm)
cset = ax.contour(x, y, posterior, zdir='y', offset=160, cmap=CM.coolwarm)
ax.set_title('Posterior')
ax.set_xlabel('gpas [uS]')
ax.set_ylabel('Ra [ohm cm]')
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/out/posterior"+str(Ra.resolution)+".png")


Ra_posterior = interpolate(Ra.values, Ra.posterior)
Ra_prior = interpolate(Ra.values, Ra.prior)
if type(Ra_posterior) is str:
    print "The posterior distribution is out of range in the direction: " + Ra_posterior
else:
    inferred_Ra = Ra_posterior[0][np.argmax(Ra_posterior[1])]
    posterior_sharpness = sharpness(Ra_posterior[0], Ra_posterior[1])
    prior_sharpness = sharpness(Ra_prior[0], Ra_prior[1])
    print "The inferred maximum probable Ra value: " + str(inferred_Ra)
    print "The sharpness of the prior distribution: " + str(prior_sharpness)
    print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
