import numpy as np
from module.simulation import stick_and_ball
from module.probability import IndependentInference, ParameterSet, RandomVariable
from module.noise import more_w_trace
from functools import partial

hz = [1, 10, 100]
duration = [3, 20, 200]

p_names = ['Ra', 'cm', 'gpas']
p_res = [10, 10, 10]  # Parameters resolution
p_range = [[40, 160], [0.4, 1.6], [0.00004, 0.00016]]  # Fixed range, but "true value" may change!
p_mean = [150., 1., 0.0001]  # Fixed prior mean
p_std = [20., 0.2, 0.00002]  # Fixed prior std

noise = 7.
noise_rep = 30
model = stick_and_ball

# Set up random seed
np.random.seed(42)


for i in range(10):
    print("\n\n--- SIMULATION FOR %ith FIXED PARAMETER ---" % (i+1))
    # Set up "true" value for this cycle
    current_value = {}                                          

    for j in range(len(p_names)):
        val = np.random.normal(p_mean[j], p_std[j])
        current_value[p_names[j]] = val


    # Set up parameters for one cycle
    current_params = []
    for idx, item in enumerate(p_names):
        current_params.append(RandomVariable(name=item, range_min=p_range[idx][0], range_max=p_range[idx][1],
                                            resolution=p_res[idx], value=current_value[item],
                                            sigma=p_std[idx], mean=p_mean[idx]))

    for item in hz:
        print("\n\n---------------------------------------- Running %i Hz zap protocol" % item)

        # Stimulus path
        stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/zap/%i/stim.txt" % item)
        working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/zaps/%i(%i)" % (item,i)

        # Generate deterministic trace and create synthetic data with noise model
        _, v = model(stype='custom', custom_stim=stim,
                     Ra=current_value['Ra'], gpas=current_value['gpas'], cm=current_value['cm'])

        # Generate noise_rep synthetic data (noise_rep portion noise realisation)
        data = more_w_trace(noise, v, noise_rep)

        pset = ParameterSet(*current_params)

        modell = partial(model, stype='custom', custom_stim=stim)
        inf = IndependentInference(model=modell, noise_std=noise, target_trace=data, parameter_set=pset, working_path=working_path)

        if __name__ == '__main__':
            inf.run_moretrace_inf()

    for item in duration:
        print("\n\n---------------------------------------- Running %i ms impulse protocol" % item)

        # Stimulus path
        stim = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/%i/stim.txt" % item)
        working_path = "/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/%i(%i)" % (item, i)

        # Generate deterministic trace and create synthetic data with noise model
        _, v = model(stype='custom', custom_stim=stim,
                     Ra=current_value['Ra'], gpas=current_value['gpas'], cm=current_value['cm'])

        # Generate n synthetic data (n noise realisation)
        data = more_w_trace(noise, v, noise_rep)

        pset = ParameterSet(*current_params)

        modell = partial(model, stype='custom', custom_stim=stim)
        inf = IndependentInference(model=modell, noise_std=noise, target_trace=data, parameter_set=pset, working_path=working_path)

        if __name__ == '__main__':
            inf.run_moretrace_inf()
