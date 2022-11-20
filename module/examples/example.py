# add project root directory to search path
import sys

import matplotlib.pyplot as plt

sys.path.append('../..')
from module.noise import white
from module.probability import RandomVariable, ParameterSet, IndependentInference
from module.simulation import stick_and_ball
from module.plot import plot_joint, fullplot

noise = 7.

# 1.) Parameters to infer
cm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=60, mean=1.2, sigma=0.2, value=1.)
gpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=60, mean=0.00008, sigma=0.00002, value=0.0001)
# Ra = RandomVariable(name='Ra', range_min=50., range_max=150., resolution=60, mean=100., sigma=20.)

# 2.) Set up parameter set
cm_gpas = ParameterSet(cm, gpas)

# 3.) Sythetic data
t, v = stick_and_ball()
exp_v = white(noise, v)

# # plot v and exp_v in t
# plt.figure()
# plt.plot(t, exp_v, label='exp_v')
# plt.plot(t, v, label='v')
# plt.legend()
# plt.show()

# 4.) Set up inference
inf = IndependentInference(model=stick_and_ball, noise_std=noise, target_trace=exp_v, parameter_set=cm_gpas,
                           working_path="/Users/admin/PROJECTS/SPE/parameter-inference/module/examples/output", speed='max', save=False)

# 5.) Run inference
if __name__ == "__main__":
    inf.run_sim()

# 6.) Run evaluation
inf.run_evaluation()

print "KL divergence test: %f" % inf.KL

# 7.) Plot solution
print inf
plot_joint(inf, cm, gpas)
fullplot(inf)