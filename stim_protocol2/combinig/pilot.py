import numpy as np
from module.simulation import stick_and_ball
from module.probability import IndependentInference, ParameterSet, RandomVariable
from module.noise import more_w_trace
from functools import partial

hz = [1, 10, 100]
duration = [3, 20, 200]

p_names = ['Ra', 'cm', 'gpas']
res = [60, 60, 60]
noise = 7.
n = 30
model = stick_and_ball

Ra = RandomVariable(name='Ra', range_min=50., range_max=150., resolution=60, mean=100., sigma=20.)
gpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=60, mean=0.0001, sigma=0.00002)
cm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=60, mean=1., sigma=0.2)

# Load fixed parameters: list of parameters to be inferred
fixed_params = [Ra, cm, gpas]

for i in range(10):
    # Set up mean and range for a cycle of simulation
    current_mean = {}
    current_minrange = []
    current_maxrange = []

    for item in fixed_params:
        if item.name == 'Ra':
            mean = np.random.normal(item.mean, item.sigma / 2.)
            current_mean['Ra'] = mean
            current_minrange.append(mean - item.offset)
            current_maxrange.append(mean + item.offset)
        else:
            mean = np.random.normal(item.mean, item.sigma)
            current_mean[item.name] = mean
            current_minrange.append(mean - item.offset)
            current_maxrange.append(mean + item.offset)

    # Biopysicsal parameters can't be negative
    for idx, item in enumerate(current_minrange):
        if item <= 0.:
            current_minrange[idx] = 0.000001

    # Set up parameters for one cycle
    current_params = []
    for idx, item in enumerate(p_names):
        current_params.append(RandomVariable(item, range_min=current_minrange[idx], range_max=current_maxrange[idx],
                                             resolution=res[idx], mean=current_mean[item],
                                             sigma=fixed_params[idx].sigma))

    for item in hz:
        print "\n\n---------------------------------------- Running %i Hz zap protocol" % item

        stim = np.loadtxt("/home/szabolcs/parameter_inference/stim_protocol2_v5/zap/%i/stim.txt" % item)
        working_path = "/home/szabolcs/parameter_inference/stim_protocol2_v5/combinig/zaps/%i(%i)" % (item,i)

        # Generate deterministic trace and create synthetic data with noise model
        _, v = model(stype='custom', custom_stim=stim,
                     Ra=current_mean['Ra'], gpas=current_mean['gpas'], cm=current_mean['cm'])

        # Generate n synthetic data (n noise realisation)
        data = more_w_trace(noise, v, n)

        pset = ParameterSet(*current_params)

        modell = partial(model, stype='custom', custom_stim=stim)
        inf = IndependentInference(model=modell, noise_std=noise, target_trace=data, parameter_set=pset, working_path=working_path)

        if __name__ == '__main__':
            inf.run_moretrace_inf()

    for item in duration:
        print "\n\n---------------------------------------- Running %i ms impulse protocol" % item

        stim = np.loadtxt("/home/szabolcs/parameter_inference/stim_protocol2_v5/steps/%i/stim.txt" % item)
        working_path = "/home/szabolcs/parameter_inference/stim_protocol2_v5/combinig/steps/%i(%i)" % (item, i)

        # Generate deterministic trace and create synthetic data with noise model
        _, v = model(stype='custom', custom_stim=stim,
                     Ra=current_mean['Ra'], gpas=current_mean['gpas'], cm=current_mean['cm'])

        # Generate n synthetic data (n noise realisation)
        data = more_w_trace(noise, v, n)

        pset = ParameterSet(*current_params)

        modell = partial(model, stype='custom', custom_stim=stim)
        inf = IndependentInference(model=modell, noise_std=noise, target_trace=data, parameter_set=pset, working_path=working_path)

        if __name__ == '__main__':
            inf.run_moretrace_inf()
