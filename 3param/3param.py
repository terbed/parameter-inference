from module.probability import RandomVariable, ParameterSet, IndependentInference
from module.plot import plot_res as plot
from module.simulation import stick_and_ball
from module.noise import white
from functools import partial

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
import matplotlib.patches as mpatches

from neuron import h, gui
import numpy as np

# --- Set Random Variables
Ra = RandomVariable(name='Ra', range_min=20., range_max=500., resolution=100, mean=100, sigma=40)
gpas = RandomVariable(name='gpas', range_min=0.00008, range_max=0.0002, resolution=80, mean=0.0001, sigma=0.00003)
cm = RandomVariable(name='cm', range_min=0.5, range_max=3, resolution=80, mean=1., sigma=0.4)

# ---- Create synthetic experimental trace with white noise
exp_noise = 6
t, v = stick_and_ball()
exp_v = white(exp_noise, v)

# ---- Debug: Big values?
_, v2 = stick_and_ball(Ra=2000., cm=2.2, Ra_max=2000., gpas=0.00024)

plt.figure()
plt.title("r: 100; 1; 0.0001;\n g: 2000; 2.2; 0.00024;")
plt.plot(t, exp_v, color='r')
plt.plot(t, v2, color='g')
plt.show()


Ra_cm = ParameterSet(Ra, cm)
Ra_gpas = ParameterSet(Ra, gpas)
cm_gpas = ParameterSet(cm, gpas)
Ra_cm_gpas = ParameterSet(Ra, cm, gpas)

inf = IndependentInference(exp_v, Ra_cm, working_path="/Users/Dani/TDK/parameter_estim/3param/Ra-cm")
inf1 = IndependentInference(exp_v, Ra_gpas, working_path="/Users/Dani/TDK/parameter_estim/3param/Ra-gpas")
inf2 = IndependentInference(exp_v, cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/3param/cm-gpas")
inf3 = IndependentInference(exp_v, Ra_cm_gpas, working_path="/Users/Dani/TDK/parameter_estim/3param/Ra-cm-gpas")

model = partial(stick_and_ball, Ra_max=Ra.range_max)  # Needed for nseg computation

if __name__ == '__main__':
    #inf.run_sim(model, exp_noise)
    #inf1.run_sim(model, exp_noise)
    #inf2.run_sim(model, exp_noise)
    inf3.run_sim(model, exp_noise)

#inf.run_evaluation()
#inf1.run_evaluation()
#inf2.run_evaluation()
inf3.run_evaluation()

# joint plots
#plot(inf, Ra, cm)
#plot(inf1, Ra, gpas)
#plot(inf2, cm, gpas)

plot(inf3, Ra, gpas)
plot(inf3, cm, gpas)
plot(inf3, Ra, cm)

# Marginal plots
#print inf
#print inf1
#print inf2
print inf3
