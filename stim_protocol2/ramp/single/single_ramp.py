import numpy as np
from module.simulation import stick_and_ball
from module.probability import RandomVariable, IndependentInference, ParameterSet
from module.noise import white
from module.plot import fullplot, plot_joint
from functools import partial

noise_sigma = 7.
stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/stim.txt")

Ra = RandomVariable(name='Ra', range_min=40., range_max=150., resolution=160, mean=100., sigma=20)
gpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=160, mean=0.0001, sigma=0.00002)
cm = RandomVariable(name='cm', range_min=0.5, range_max=1.7, resolution=160, mean=1., sigma=0.2)

# Generate deterministic trace and create synthetic data with noise model
# t, v = stick_and_ball(stype='custom', custom_stim=stim)
# data = white(noise_sigma, v)
# np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/single/data(9).txt", data)

data = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/single/data(9).txt")

multi_comp = partial(stick_and_ball, stype='custom', custom_stim=stim)  # fix chosen stimulus type for simulations

Ra_cm_gpas = ParameterSet(Ra, cm, gpas)
inference = IndependentInference(model=multi_comp, noise_std=noise_sigma, target_trace=data, parameter_set=Ra_cm_gpas,
                                 working_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/single", speed="min")

if __name__ == '__main__':
    inference.run_sim()

# inference.save_result()
inference.run_evaluation()

print(inference)
fullplot(inference)
plot_joint(inference, Ra, gpas)
plot_joint(inference, Ra, cm)
plot_joint(inference, cm, gpas)