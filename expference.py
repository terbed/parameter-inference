"""
Inference on true experimental data
"""

from module.likelihood import dependent_2d as likelihood_func
from module.prior import normal2d
from module.trace import sharpness, interpolate
from module.noise import colored

from neuron import h, gui
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def experiment_sim(Ra, gpas, Ra_max=150, dt=0.1):

    # Load morphology
    h('load_file("/Users/Dani/TDK/parameter_estim/exp/morphology_131117-C2.hoc")')

    # Print information
    h.psection()

    # Set the appropriate "nseg"
    for sec in h.allsec():
        sec.Ra = Ra_max
    h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

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
# Colored noise parameters
D = 0.0001
lamb = 0.1
dt = 0.1

Ra = 157.362128223           # parameter optimisation algorithm found this
g_pas = 0.000403860792541     # parameter optimisation algorithm found this

gpas_min = 0.0003
gpas_max = 0.0005
gpas_num = 100.
gpas_values = np.linspace(gpas_min, gpas_max, gpas_num)
gpas_range = gpas_max - gpas_min
gpas_step = gpas_range / gpas_num

Ra_min = 57.
Ra_max = 257.
Ra_num = 100.
Ra_values = np.linspace(Ra_min, Ra_max, Ra_num)
Ra_range = Ra_max - Ra_min
Ra_step = Ra_range / Ra_num

Ra_sig = 20
Ra_mean = 157.362128223      # parameter optimisation algorithm found this
gpas_sig = 0.00002
gpas_mean = 0.000403860792541     # parameter optimisation algorithm found this


# Load experimental trace
experimental_trace = genfromtxt("/Users/Dani/TDK/parameter_estim/exp/resampled_experimental_trace")
plt.figure()
plt.title("Experimental trace")
plt.xlabel("Time [ns]")
plt.ylabel("Voltage [mV]")
plt.plot(experimental_trace[:, 0], experimental_trace[:, 1])
# plt.savefig("/Users/Dani/TDK/parameter_estim/exp/experimental_trace.png")


# Check simulation trace
t_sim, v_sim = experiment_sim(Ra, g_pas)
v_sim = colored(0.0001, 0.1, 0.1, v_sim)
plt.figure()
plt.title("Simulation trace")
plt.xlabel("Time [ns]")
plt.ylabel("Voltage [mV]")
plt.plot(t_sim, v_sim)
# plt.savefig("/Users/Dani/TDK/parameter_estim/exp/simulated_trace" + str(Ra) + "_gpas" + str(g_pas) + ".png")

plt.figure()
plt.title("Together")
plt.plot(experimental_trace[:, 0], experimental_trace[:, 1], 'r')
plt.plot(t_sim, v_sim, 'g')
# plt.savefig("/Users/Dani/TDK/parameter_estim/exp/exp_and_sim_trace_Ra" + str(Ra) + "_gpas" + str(g_pas) + ".png")
plt.show()

t = experimental_trace[:, 0]
exp_v = experimental_trace[:, 1]

# Load inverse covariant matrix - [Generate inverse covariant matrix]
invcovmat = genfromtxt('/Users/Dani/TDK/parameter_estim/exp/inv_covmat_0.0001_0.1.txt')

# TRY TO INFER BACK Ra PARAMETER
likelihood = likelihood_func(experiment_sim, Ra_values, gpas_values, invcovmat, exp_v)

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
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/Ra_posterior"+str(Ra_num)+".png")


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
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/likelihood"+str(Ra_num)+".png")


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
plt.savefig("/Users/Dani/TDK/parameter_estim/exp/posterior"+str(Ra_num)+".png")


Ra_posterior = interpolate(Ra_values, Ra_posterior)
if type(Ra_posterior) is str:
    print "The posterior distribution is out of range in the direction: " + Ra_posterior
else:
    inferred_Ra = Ra_values[np.argmax(Ra_posterior)]
    posterior_sharpness = sharpness(Ra_values, Ra_posterior)
    prior_sharpness = sharpness(Ra_values, Ra_prior)
    print "The inferred maximum probable Ra value: " + str(inferred_Ra)
    print "The sharpness of the prior distribution: " + str(prior_sharpness)
    print "The sharpness of the posterior distribution: " + str(posterior_sharpness)

plt.show()
