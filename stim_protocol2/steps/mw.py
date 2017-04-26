import numpy as np
from module.simulation import stick_and_ball
from module.probability import RandomVariable, IndependentInference, ParameterSet
from module.noise import white
from module.trace import stat
from module.plot import plot_stat
from functools import partial
from matplotlib import pyplot as plt
import time

num_of_iter = 50

noise_sigma = 7.

Ra_stat = np.zeros((num_of_iter, 5), dtype=np.float)
gpas_stat = np.zeros((num_of_iter, 5), dtype=np.float)
cm_stat = np.zeros((num_of_iter, 5), dtype=np.float)

# Only for plotting
pRa = RandomVariable(name='Ra', range_min=50, range_max=150, resolution=60, mean=100, sigma=20)
pgpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=60, mean=0.0001, sigma=0.00002)
pcm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=60, mean=1., sigma=0.2)

startTime = time.time()
for i in range(num_of_iter):
    print str(i) + " is DONE out of " + str(num_of_iter)

    # Sampling current parameter from normal distribution
    current_Ra = np.random.normal(pRa.value, 10.)
    current_gpas = np.random.normal(pgpas.value, 0.00002)
    current_cm = np.random.normal(pcm.value, 0.1)

    # Generate deterministic trace and create synthetic data with noise model
    t, v = stick_and_ball(Ra=current_Ra, gpas=current_gpas, cm=current_cm, stype='steps')
    data = white(noise_sigma, v)

    # if i == 0:
    #     plt.figure()
    #     plt.title("Neuron voltage response to stimuli")
    #     plt.xlabel('Time [ms]')
    #     plt.ylabel('Voltage [mV]')
    #     plt.plot(t, v, color='#2FA5A0')
    #     plt.show()

    # Set up range in a way that the true parameter value will be in the middle
    Ra_start = current_Ra - 50
    Ra_end = current_Ra + 50
    gpas_start = current_gpas - 0.00005
    gpas_end = current_gpas + 0.00005
    cm_start = current_cm - 0.5
    cm_end = current_cm + 0.5

    if Ra_start <= 0:  # ValueError: Ra must be > 0.
        Ra_start = 1
    if gpas_start <= 0:  # ValueError: Ra must be > 0.
        gpas_start = 0.0000001
    if cm_start <= 0:
        cm_start = 0.0001

    # Set up random variables
    Ra = RandomVariable(name='Ra', range_min=Ra_start, range_max=Ra_end, resolution=40, mean=current_Ra, sigma=pRa.sigma)
    gpas = RandomVariable(name='gpas', range_min=gpas_start, range_max=gpas_end, resolution=40, mean=current_gpas, sigma=pgpas.sigma)
    cm = RandomVariable(name='cm', range_min=cm_start, range_max=cm_end, resolution=40, mean=current_cm, sigma=pcm.sigma)

    Ra_cm_gpas = ParameterSet(Ra, cm, gpas)
    inference = IndependentInference(data, Ra_cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/steps")

    multi_comp = partial(stick_and_ball, stype='steps')  # fix chosen stimulus type for simulations

    if __name__ == '__main__':
        inference.run_sim(multi_comp, noise_sigma)

    inference.run_evaluation()

    # Do statistics for the current inference
    Ra_stat[i, 0], Ra_stat[i, 1], Ra_stat[i, 2], Ra_stat[i, 3], Ra_stat[4] = stat(Ra)
    gpas_stat[i, 0], gpas_stat[i, 1], gpas_stat[i, 2], gpas_stat[i, 3], gpas_stat[4] = stat(gpas)
    cm_stat[i, 0], cm_stat[i, 1], cm_stat[i, 2], cm_stat[i, 3], cm_stat[4] = stat(cm)


runningTime = (time.time() - startTime) / 60
lasted = "The Ra-gpas-cm ball-and-stick simulation was running for %f minutes\n" % runningTime
configuration = "MacBook Pro (Retina, Mid 2012); 2.6 GHz Intel Core i7; 8 GB 1600 MHz DDR3; macOS Sierra 10.12.4\n"
setup1 = 'Multi compartment simulation; White noise sigma=7; steps stimulus; Ra parameter; dt=0.1\n'
setup2 = 'Multi compartment simulation; White noise sigma=7; steps stimulus; gpas parameter; dt=0.1\n'
setup3 = 'Multi compartment simulation; White noise sigma=7; steps stimulus; cm parameter; dt=0.1\n'
header1 = "Number of simulations: " + str(num_of_iter) + '\n' + setup1 + configuration + lasted
header2 = "Number of simulations: " + str(num_of_iter) + '\n' + setup2 + configuration + lasted
header3 = "Number of simulations: " + str(num_of_iter) + '\n' + setup3 + configuration + lasted

# Save out statistic to file for occurent later analysis
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/steps/Ra_stat.txt', X=Ra_stat,
           header=header1 + 'sigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/steps/gpas_stat.txt', X=gpas_stat,
           header=header2 + '\nsigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/steps/cm_stat.txt', X=cm_stat,
           header=header3 + '\nsigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')

# Plot statistics
plot_stat(Ra_stat, pRa, path='/Users/Dani/TDK/parameter_estim/stim_protocol2/steps')
plot_stat(gpas_stat, pgpas, path='/Users/Dani/TDK/parameter_estim/stim_protocol2/steps')
plot_stat(cm_stat, pcm, path='/Users/Dani/TDK/parameter_estim/stim_protocol2/steps')

