from neuron import h, gui
from matplotlib import pyplot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def normalDistributionValueGenerator(mean, sig, x):
    normal_distribution_value = 1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mean)**2/(2*sig**2))
    return normal_distribution_value


def priorGenerator(mean, sig, values):
    prior = []
    for x in values:
        prior.append(normalDistributionValueGenerator(mean, sig, x))
    
    return prior

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
    h.finitialize(-70)             # Starting membrane potential ?
    h.run()
    
    return (v_vec, t_vec)


# Load experimental voltage trace
sigma = 7
experimental_trace = np.loadtxt('exp_trace.txt')

# Terminal input
ns = raw_input('Resolution number: ')


## PARAMETERS

# Parameters for prior distribution
gpas_mean = 0.0001
gpas_sig = 0.00002
cm_sig = 0.2
cm_mean = 1.0

gpas_min = 0.00001
gpas_max = 0.00019
gpas_num = float(ns)
gpas_step = (gpas_max - gpas_min)/gpas_num
gpas_values = np.random.normal(gpas_mean, gpas_sig, gpas_num)
gpas_values = np.sort(gpas_values)

cm_min = 0.3
cm_max = 1.7
cm_num = 200.0
cm_step = (cm_max - cm_min)/cm_num
cm_values = np.linspace(cm_min, cm_max, cm_num)


## MAXIMUM LIKELIHOOD METHOD
# Create likelihood

def logLikelihood2D(x_values, y_values, sig, target_trace):
    x_y = np.zeros( (len(x_values), len(y_values)) )
    summed_square_devitation2D =  np.zeros( (len(x_values), len(y_values)))
    
    # Fill 2D likelihood matrix with log_likelihood elements
    for i in range( len(x_values) ):
        for j in range( len(y_values) ):
            v_vec = h.Vector()
            (v_vec, _) = OneCompartmentSimulation(x_values[i], y_values[j])
            
            v_curr = v_vec.to_python() # Convert neuron vector to numpy vector
            
            v_dev = np.subtract(target_trace, v_curr)
            v_square_dev = (v_dev**2)
            v_summed_square_dev = np.sum(v_square_dev)
            
            summed_square_devitation2D[i,j] = v_summed_square_dev
            log_likelihood = - v_summed_square_dev/(2*sig**2)
            
            x_y[i,j] = log_likelihood

    # "Normalize" 2D log_likelihood for numeric reasons
    x_y = np.subtract(x_y, np.amax(x_y))
    
    pyplot.figure()
    pyplot.title("Raw 2D square devitation")
    pyplot.imshow(summed_square_devitation2D)
    pyplot.grid(True)
    pyplot.xlabel('gpas')
    pyplot.ylabel('cm')
    
    return x_y



log_likelihood_2d = logLikelihood2D(cm_values, gpas_values, sigma, experimental_trace)
likelihood2d = np.exp(log_likelihood_2d)

# Plot
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.set_title('likelihood')
pyplot.xlabel('gpas')
pyplot.ylabel('cm')
x, y = np.meshgrid(gpas_values, cm_values)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, likelihood2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)


# Marginlaize for cm
likelihood = np.sum(likelihood2d, axis = 1) / gpas_num

# Prior distribution for cm
prior = priorGenerator(cm_mean, cm_sig, cm_values)
likelihood = np.multiply(likelihood, prior)

# Posterior distribution
norm = np.sum(likelihood) / cm_num
posterior = likelihood/norm


pyplot.figure()
pyplot.title("Posterior distribution")
pyplot.plot(cm_values, posterior)
pyplot.xlabel('cm')
pyplot.ylabel('probability')

pyplot.show()

np.savetxt('data2/posterior_' + ns + 'p' + '.txt', np.c_[cm_values, posterior], delimiter = ' ')
