import numpy as np
from module.simulation import stick_and_ball
from module.probability import IndependentInference, ParameterSet, RandomVariable
from module.noise import white
from functools import partial

hz = [0, 1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 150]

p_names = ['Ra', 'cm', 'gpas']
speed = 'min'
noise = 7.
n = 30
model = stick_and_ball

Ra = RandomVariable(name='Ra', range_min=50., range_max=150., resolution=60, mean=100., sigma=20.)
gpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=60, mean=0.0001, sigma=0.00002)
cm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=60, mean=1., sigma=0.2)

for item in hz:
    print "\n\n---------------------------------------- Running %i Hz zap protocol" % item

    stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/%i/stim.txt" % item)
    working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/zaps/%i" % item

    # Load fixed parameters: list of parameters to be inferred
    fixed_params = [Ra, cm, gpas]

    # Generate deterministic trace and create synthetic data with noise model
    t, v = model(stype='custom', custom_stim=stim)
    data = white(noise, v)

    pset = ParameterSet(*fixed_params)

    modell = partial(model, stype='custom', custom_stim=stim)
    inf = IndependentInference(model=modell, noise_std=noise, target_trace=data, parameter_set=pset, working_path=working_path, speed=speed)

    if __name__ == '__main__':
        inf.run_sim()

    inf.run_evaluation()