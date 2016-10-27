from neuron import h, gui
from matplotlib import pyplot
import numpy as np

def deterministicTrace(x,y):
	# Create deterministic trace
	v_vec = h.Vector()
	t_vec = h.Vector()
	(v, t) = OneCompartmentSimulation(x,y)

	return (t,v)

def whiteNoise(mu, sigma, v, numOfPoints):
	noise_signal = np.random.normal(mu, sigma, numOfPoints)
	exp_v = np.add(v,noise_signal)

	return exp_v

def OneCompartmentSimulation(cm, g_pas):
	# Creating one compartment passive  modell (interacting with neuron)
	soma = h.Section(name = 'soma')
	soma.L = soma.diam = 50  # it is a sphere
	soma.v = -65
	soma.cm = cm

	# Insert passieve conductance
	soma.insert('pas')
	soma.g_pas = g_pas
	soma.e_pas = -65

	# Creating stimulus
	stim = h.IClamp(soma(0.5))
	stim.delay = 30
	stim.amp = 0.1
	stim.dur = 100

	# Print Information
	h.psection()

	# Set up recording Vectors
	v_vec = h.Vector()             # Membrane potential vector
	t_vec = h.Vector()             # Time stamp vector
	v_vec.record(soma(0.5)._ref_v)
	t_vec.record(h._ref_t)

	# Simulation duration and RUN
	h.tstop = 200                  # Simulation end
	h.dt = 0.025                   # Time step (iteration)
	h.finitialize(-70)             # Starting membrane potential ??? Not workin
	h.run()
	
	v = v_vec.to_python()
	t = t_vec.to_python()

	return (v, t)

# TARGET TRACE ------------------------------------------------------
# "real" values for c_m and gpas
cm = 1
g_pas = 0.0001

# Creating target trace
(t,v) = deterministicTrace(cm, g_pas)

mu = 0
sigma = 7

target_trace = whiteNoise(mu, sigma, v, len(v))

# Plot the result (Trace with noise)
pyplot.figure()
pyplot.xlabel('time (ms)')
pyplot.ylabel('exp_V (mV)')
pyplot.plot(t, target_trace)
pyplot.show()

np.savetxt("exp_trace.txt", target_trace) 
