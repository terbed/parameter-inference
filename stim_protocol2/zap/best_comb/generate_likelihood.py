import numpy as np
from module.simulation import stick_and_ball
from module.probability import IndependentInference, ParameterSet
from module.noise import white
from module.load import get_default_param
from functools import partial


hz = [10, 75, 150]

p_names = ['Ra', 'cm', 'gpas']
speed = 'min'
noise = 7.
n = 50
model = stick_and_ball

for item in hz:
    print "\n\n---------------------------------------- Running %i ms impulse protocol" % item

    stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/best_comb/%i/stim.txt" % item)
    working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/best_comb/%i" % item

    # Do statistics for each parameter
    stat_list = []
    for _ in p_names:
        stat_list.append(np.empty((n, 6), dtype=np.float))

    # Load fixed parameters: list of parameters to be inferred
    fixed_params = []
    for name in p_names:
        fixed_params.append(get_default_param(name))

    # Generate deterministic trace and create synthetic data with noise model
    t, v = model(stype='custom', custom_stim=stim)
    data = white(noise, v)

    pset = ParameterSet(*fixed_params)

    modell = partial(model, stype='custom', custom_stim=stim)
    inf = IndependentInference(model=modell, noise_std=noise, target_trace=data, parameter_set=pset, working_path=working_path, speed=speed)

    if __name__ == '__main__':
        inf.run_sim()

