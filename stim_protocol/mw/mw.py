import numpy as np
from module.simulation import stick_and_ball
from module.probability import RandomVariable, IndependentInference, ParameterSet
from module.noise import white
from module.trace import stat
from module.plot import plot_stat
from functools import partial
from matplotlib import pyplot as plt
import time

num_of_iter = 100

noise_sigma = 7.

Ra_stat = np.zeros((num_of_iter, 5), dtype=np.float)
gpas_stat = np.zeros((num_of_iter, 5), dtype=np.float)

# Only for plotting
pRa = RandomVariable(name='Ra', range_min=50, range_max=150, resolution=80, mean=100, sigma=20)
pgpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=80, mean=0.0001, sigma=0.00002)

startTime = time.time()
for i in range(num_of_iter):
    print str(i) + " is DONE out of " + str(num_of_iter)

    # Sampling current parameter from normal distribution
    current_Ra = np.random.normal(pRa.value, pRa.sigma)
    current_gpas = np.random.normal(pgpas.value, pgpas.sigma)

    # Generate deterministic trace and create synthetic data with noise model
    t, v = stick_and_ball(Ra=current_Ra, gpas=current_gpas, stype='narrow')
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
    # TODO crate log-normal prior distribution for Ra
    if Ra_start < 0:  # ValueError: Ra must be > 0.
        Ra_start = 1

    gpas_start = current_gpas - 0.00005
    gpas_end = current_gpas + 0.00005

    # Set up random variables
    Ra = RandomVariable(name='Ra', range_min=Ra_start, range_max=Ra_end, resolution=100, mean=current_Ra, sigma=pRa.sigma)
    gpas = RandomVariable(name='gpas', range_min=gpas_start, range_max=gpas_end, resolution=100, mean=current_gpas, sigma=pgpas.sigma)

    Ra_gpas = ParameterSet(Ra, gpas)
    inference = IndependentInference(data, Ra_gpas)
    multi_comp = partial(stick_and_ball, stype='narrow')  # fix chosen stimulus type for simulations

    if __name__ == '__main__':
        inference.run_sim(multi_comp, noise_sigma)

    inference.run_evaluation()

    # Do statistics for the current inference
    if stat(Ra) is not str:
        Ra_stat[i, 0], Ra_stat[i, 1], Ra_stat[i, 2], Ra_stat[i, 3], Ra_stat[4] = stat(Ra)
    else:
        print "\n WARNING!!! OUT OF RANGE!!!  You should delete the simulation data lines with no results! (0 values)\n"
    if stat(gpas) is not str:
        gpas_stat[i, 0], gpas_stat[i, 1], gpas_stat[i, 2], gpas_stat[i, 3], gpas_stat[4] = stat(gpas)
    else:
        print "\n WARNING!!! OUT OF RANGE!!!  You should delete the simulation data lines with no results! (0 values)\n"

runningTime = (time.time() - startTime) / 60
lasted = "The simulation was running for %f minutes\n" % runningTime
configuration = "MacBook Pro (Retina, Mid 2012); 2.6 GHz Intel Core i7; 8 GB 1600 MHz DDR3; macOS Sierra 10.12.1\n"
setup1 = 'Multi compartment simulation; White noise sigma=7; narrow stimulus; Ra parameter; dt=0.1\n'
setup2 = 'Multi compartment simulation; White noise sigma=7; narrow stimulus; gpas parameter; dt=0.1\n'
header1 = "Number of simulations: " + str(num_of_iter) + '\n' + setup1 + configuration + lasted
header2 = "Number of simulations: " + str(num_of_iter) + '\n' + setup2 + configuration + lasted

# Save out statistic to file for occurent later analysis
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol/mw/narrow/Ra_stat.txt', X=Ra_stat,
           header=header1 + 'sigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol/mw/narrow/gpas_stat.txt', X=gpas_stat,
           header=header2 + '\nsigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')

# Plot statistics
plot_stat(Ra_stat, pRa, path='/Users/Dani/TDK/parameter_estim/stim_protocol/mw/narrow')
plot_stat(gpas_stat, pgpas, path='/Users/Dani/TDK/parameter_estim/stim_protocol/mw/narrow')

