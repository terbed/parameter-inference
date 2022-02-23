import numpy as np
from module.simulation import one_compartment
from module.probability import RandomVariable, IndependentInference, ParameterSet
from module.noise import white
from module.trace import stat
from module.plot import plot_stat
from functools import partial
from matplotlib import pyplot as plt
import time

num_of_iter = 100

cm_stat = np.zeros((num_of_iter, 5), dtype=np.float)
gpas_stat = np.zeros((num_of_iter, 5), dtype=np.float)

cm_mean = 1.
cm_sig = 0.2
gpas_mean = 0.0001
gpas_sig = 0.00002
noise_sigma = 7.

cm_start = 0.5
cm_end = 1.5

gpas_start = gpas_mean - 0.00005
gpas_end = gpas_mean + 0.00005

# Only for plotting
pcm = RandomVariable(name='cm', range_min=cm_start, range_max=cm_end, resolution=100, mean=cm_mean, sigma=cm_sig)
pgpas = RandomVariable(name='gpas', range_min=gpas_start, range_max=gpas_end, resolution=100, mean=gpas_mean,
                      sigma=gpas_sig)

startTime = time.time()
for i in range(num_of_iter):
    print(str(i) + " is DONE out of " + str(num_of_iter))

    # Sampling current parameter from normal distribution
    current_cm = np.random.normal(cm_mean, cm_sig)
    current_gpas = np.random.normal(gpas_mean, gpas_sig)

    # Generate deterministic trace and create synthetic data with noise model
    t, v = one_compartment(cm=current_cm, gpas=current_gpas, stype='steps')
    data = white(noise_sigma, v)

    # if i == 0:
    #     plt.figure()
    #     plt.title("Neuron voltage response to stimuli")
    #     plt.xlabel('Time [ms]')
    #     plt.ylabel('Voltage [mV]')
    #     plt.plot(t, v, color='#2FA5A0')
    #     plt.show()
    #     plt.savefig('/Users/Dani/TDK/parameter_estim/stim_protocol/steps_stimuli_type.png')

    # Set up range in a way that the true parameter value will be in the middle
    cm_start = current_cm - 0.5
    cm_end = current_cm + 0.5

    gpas_start = current_gpas - 0.00005
    gpas_end = current_gpas + 0.00005

    # Set up random variables
    cm = RandomVariable(name='cm', range_min=cm_start, range_max=cm_end, resolution=100, mean=current_cm, sigma=cm_sig)
    gpas = RandomVariable(name='gpas', range_min=gpas_start, range_max=gpas_end, resolution=100, mean=current_gpas, sigma=gpas_sig)

    cm_gpas = ParameterSet(cm, gpas)
    inference = IndependentInference(data, cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/stim_protocol/ow/steps")
    one_comp = partial(one_compartment, stype='steps')  # fix chosen stimulus type for simulations

    if __name__ == '__main__':
        inference.run_sim(one_comp, noise_sigma)

    inference.run_evaluation()

    # Do statistics for the current inference
    if stat(cm) is not str:
        cm_stat[i, 0], cm_stat[i, 1], cm_stat[i, 2], cm_stat[i, 3], cm_stat[4] = stat(cm)
    else:
        print("\n WARNING!!! OUT OF RANGE!!!  You should delete the simulation data lines with no results! (0 values)\n")
    if stat(gpas) is not str:
        gpas_stat[i, 0], gpas_stat[i, 1], gpas_stat[i, 2], gpas_stat[i, 3], gpas_stat[4] = stat(gpas)
    else:
        print("\n WARNING!!! OUT OF RANGE!!!  You should delete the simulation data lines with no results! (0 values)\n")

runningTime = (time.time() - startTime) / 60
lasted = "The simulation was running for %f minutes\n" % runningTime
configuration = "2.6 GHz Intel Core i7; 8 GB 1600 MHz DDR3; macOS Sierra 10.12.1\n"
setup1 = 'One compartment simulation; White noise sigma=7; steps stimulus; cm parameter; dt=0.1\n'
setup2 = 'One compartment simulation; White noise sigma=7; steps stimulus; gpas parameter; dt=0.1\n'
header1 = "Number of simulations: " + str(num_of_iter) + '\n' + setup1 + configuration + lasted
header2 = "Number of simulations: " + str(num_of_iter) + '\n' + setup2 + configuration + lasted

# Save out statistic to file for occurent later analysis
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol/ow/steps/cm_stat.txt', X=cm_stat,
           header=header1 + 'sigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')
np.savetxt(fname='/Users/Dani/TDK/parameter_estim/stim_protocol/ow/steps/gpas_stat.txt', X=gpas_stat,
           header=header2 + '\nsigma\tdiff\taccuracy\tsharper\tsigma_err', delimiter='\t')

# Plot statistics
plot_stat(cm_stat, pcm, path='/Users/Dani/TDK/parameter_estim/stim_protocol/ow/steps')
plot_stat(gpas_stat, pgpas, path='/Users/Dani/TDK/parameter_estim/stim_protocol/ow/steps')
