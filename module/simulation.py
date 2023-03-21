from neuron import h, gui
from module.probability import RandomVariable
import numpy as np

def get_default_param(name):
    Ra = RandomVariable(name='Ra', range_min=50., range_max=150., resolution=10, mean=100., sigma=20.)
    gpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=10, mean=0.0001, sigma=0.00002)
    cm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=10, mean=1., sigma=0.2)
    dict = {'Ra' : Ra, 'cm': cm, 'gpas': gpas}

    return dict[name]


def one_compartment(cm=1., gpas=0.0001, dt=0.1, stype='broad', custom_stim=None):
    """ One compartment simulation, variables: membrane capacitance and passive conductance """
    # Creating one compartment passive  model (interacting with neuron)
    soma = h.Section(name='soma')
    soma.L = soma.diam = 50                             # it is a sphere
    soma.v = -65
    soma.cm = cm                                        # parameter to infer

    # Insert passive conductance
    soma.insert('pas')
    soma.g_pas = gpas                                  # parameter to infer
    soma.e_pas = -65

    h.dt = dt                                           # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Creating stimulus
    # Here we define three kind of experimental protocol:
    # 1.) broad electrode current
    # 2.) narrow electrode current
    # 3.) both
    if stype == 'broad':
        h.tstop = 300
        stim = h.IClamp(soma(0.5))
        stim.delay = 20
        stim.amp = 0.1
        stim.dur = 175
    elif stype == 'narrow':
        h.tstop = 100
        stim = h.IClamp(soma(0.5))
        stim.delay = 10
        stim.amp = 0.5
        stim.dur = 5
    elif stype == 'both':
        h.tstop = 400
        stim1 = h.IClamp(soma(0.5))
        stim1.delay = 10
        stim1.amp = 0.5
        stim1.dur = 5

        stim2 = h.IClamp(soma(0.5))
        stim2.delay = 120
        stim2.amp = 0.1
        stim2.dur = 175
    elif stype == 'steps':
        h.tstop = 500
        stim1 = h.IClamp(soma(0.5))
        stim1.delay = 10
        stim1.amp = 0.5
        stim1.dur = 5

        stim2 = h.IClamp(soma(0.5))
        stim2.delay = 120
        stim2.amp = 0.1
        stim2.dur = 175

        stim3 = h.IClamp(soma(0.5))
        stim3.delay = 400
        stim3.amp = 0.35
        stim3.dur = 5
    elif stype == 'custom':
        h.tstop = len(custom_stim)*dt
        h.load_file("vplay.hoc")
        vec = h.Vector(custom_stim)
        istim = h.IClamp(soma(0.5))
        vec.play(istim._ref_amp, h.dt)
        istim.delay = 0   # Just for Neuron
        istim.dur = 1e9   # Just for Neuron

    # Print Information
    # h.psection()

    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()                                  # Membrane potential vector
    t_vec = h.Vector()                                  # Time stamp vector
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation  RUN
    h.v_init = -65
    h.finitialize(h.v_init)                             # Starting membrane potential

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return np.array(t), np.array(v)


def stick_and_ball(Ra=100., gpas=0.0001, cm=1., Ra_max=250., dt=0.1, stype='both', custom_stim=None):
    """ Stick and Ball model variables: Passive conductance and axial resistance """
    # Create Sections
    soma = h.Section(name='soma')
    dend = h.Section(name='dend')

    # Topology
    dend.connect(soma(1))

    # Geometry
    soma.L = soma.diam = 30
    dend.L = 1000
    dend.diam = 3

    # Simulation duration and RUN
    h.dt = dt                                                       # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Set the appropriate "nseg"
    for sec in h.allsec():
        sec.Ra = Ra_max
    h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra                                                 # Ra is a parameter to infer
        sec.cm = cm
        sec.v = -65

        sec.insert('pas')
        sec.g_pas = gpas                                            # gpas is a parameter to infer
        sec.e_pas = -65

    # Stimulus
    # Here we define three kind of experimental protocol:
    # 1.) brad electrode current
    # 2.) narrow electrode current
    # 3.) both
    if stype == 'broad':
        h.tstop = 300
        stim = h.IClamp(soma(0.5))
        stim.delay = 20
        stim.amp = 0.1
        stim.dur = 175
    elif stype == 'narrow':
        h.tstop = 100
        stim = h.IClamp(soma(0.5))
        stim.delay = 10
        stim.amp = 0.5
        stim.dur = 5
    elif stype == 'both':
        h.tstop = 400
        stim1 = h.IClamp(soma(0.5))
        stim1.delay = 10
        stim1.amp = 0.5
        stim1.dur = 5

        stim2 = h.IClamp(soma(0.5))
        stim2.delay = 120
        stim2.amp = 0.1
        stim2.dur = 175
    elif stype == 'steps':
        h.tstop = 500
        stim1 = h.IClamp(soma(0.5))
        stim1.delay = 10
        stim1.amp = 0.5
        stim1.dur = 5

        stim2 = h.IClamp(soma(0.5))
        stim2.delay = 120
        stim2.amp = 0.1
        stim2.dur = 175

        stim3 = h.IClamp(soma(0.5))
        stim3.delay = 400
        stim3.amp = 0.35
        stim3.dur = 5
    elif stype == 'custom':
        h.tstop = len(custom_stim)*dt
        h.load_file("vplay.hoc")
        vec = h.Vector(custom_stim)
        istim = h.IClamp(soma(0.5))
        vec.play(istim._ref_amp, h.dt)
        istim.delay = 0   # Just for Neuron
        istim.dur = 1e9   # Just for Neuron

    # Run simulation ->
    # Print information
    # h.psection()

    # Set up recording Vectors
    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    h.v_init = -65
    h.finitialize(h.v_init)                                         # Starting membrane potential

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return np.array(t), np.array(v)


def exp_model(Ra=157.3621, gpas=0.000403860792, cm=7.849480, dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    #h.psection()

    # Stimulus
    stim1 = h.IClamp(h.soma(0.01))
    stim1.delay = 200
    stim1.amp = 0.5
    stim1.dur = 2.9

    stim2 = h.IClamp(h.soma(0.01))
    stim2.delay = 503
    stim2.amp = 0.01
    stim2.dur = 599.9

    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.tstop = (15000-1)*dt  # Simulation end
    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def exp_model2(Ra=157.3621, gpas=0.000403860792, cm=7.849480, dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    #h.psection()

    # Stimulus
    stim1 = h.IClamp(h.soma(0.01))
    stim1.delay = 199.9
    stim1.amp = 0.5
    stim1.dur = 3.1

    stim2 = h.IClamp(h.soma(0.01))
    stim2.delay = 503
    stim2.amp = 0.01
    stim2.dur = 599.9

    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.tstop = (15000-1)*dt  # Simulation end
    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def corrected_experimental_protocol_with_linear_gpas_distr(Ra=157.3621, gpas_soma=0.000403860792, k=0.001, cm=3., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas_soma  # gpas is a parameter to infer
        for seg in sec:
            h('soma distance()')
            dist = (h.distance(seg.x))
            gpas = gpas_soma * (1 + k * dist)

            if gpas < 0:
                seg.g_pas = 0
                print("WARNING!!! 'gpas' is in negative! Corrected to zero.")
            else:
                seg.g_pas = gpas

        sec.e_pas = 0

    # Print information
    #h.psection()

    # Stimulus
    stim1 = h.IClamp(h.soma(0.01))
    stim1.delay = 199.9
    stim1.amp = 0.5
    stim1.dur = 3.1

    stim2 = h.IClamp(h.soma(0.01))
    stim2.delay = 503
    stim2.amp = 0.01
    stim2.dur = 599.9

    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.tstop = (15000-1)*dt  # Simulation end
    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model(stim, gpas=0.0001, Ra=100., cm=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_2(stim, v_init=-69.196, stim_sec=93, stim_pos=0.5, gpas=0.0001, Ra=100., ffact=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = 1   # parameter optimisation algorithm found this
        sec.v =v_init

        sec.insert('pas')
        for seg in sec:
            seg.g_pas = gpas  # gpas is a parameter to infer
            seg.e_pas = v_init

    for sec in h.basal:
        sec.cm *= ffact
        for seg in sec:
            seg.g_pas *= ffact

    for sec in h.apical:
        sec.cm *= ffact
        for seg in sec:
            seg.g_pas *= ffact

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.apic[stim_sec](stim_pos))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.apic[stim_sec](stim_pos)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = v_init
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_3(stim, v_init=-65, gpas=0.0001, Ra=100., cm=1., ffact=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra   # Ra is a parameter to infer
        sec.cm = cm   # cm parameter to infer
        sec.v = v_init

        sec.insert('pas')
        for seg in sec:
            seg.g_pas = gpas  # gpas is a parameter to infer
            seg.e_pas = v_init

    for sec in h.basal:
        sec.cm *= ffact
        for seg in sec:
            seg.g_pas *= ffact

    for sec in h.apical:
        sec.cm *= ffact
        for seg in sec:
            seg.g_pas *= ffact

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = v_init
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_dend(stim, gpas=0.0001, Ra=100., cm=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.apic[30](0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.apic[30](0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_sdend_rsoma(stim, gpas=0.0001, Ra=100., cm=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.apic[30](0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_ssoma_rdend(stim, gpas=0.0001, Ra=100., cm=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.apic[30](0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_srsoma_rdend(stim, gpas=0.0001, Ra=100., cm=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas  # gpas is a parameter to infer
        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    vs_vec = h.Vector()  # Membrane potential vector
    vd_vec = h.Vector()
    t_vec = h.Vector()  # Time stamp vector
    vd_vec.record(h.apic[30](0.5)._ref_v)
    vs_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v_soma = vs_vec.to_python()
    v_dend = vd_vec.to_python()

    v_soma = np.array(v_soma)
    v_dend = np.array(v_dend)

    v = np.concatenate((v_soma, v_dend))

    return t, v


def real_morphology_model_soma_spatial(stim, k=0.001, gpas_soma=0.0001, Ra=100., cm=1., dt=0.1):
    """
    asdbkjas

    :param stim:
    :param k:
    :param gpas_soma:
    :param Ra:
    :param cm:
    :param dt:
    :return:
    """
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas_soma  # gpas is a parameter to infer

        for seg in sec:
            h('soma distance()')
            dist = (h.distance(seg.x))
            gpas = gpas_soma*(1+k*dist)

            if gpas < 0:
                seg.g_pas = 0
                print("WARNING!!! 'gpas' is in negative! Corrected to zero.")
            else:
                seg.g_pas = gpas

        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_dend_spatial(stim, d=30, k=0.001, gpas_soma=0.0001, Ra=100., cm=1., dt=0.1):
    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm   # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas_soma  # gpas is a parameter to infer

        for seg in sec:
            h('soma distance()')
            dist = (h.distance(seg.x))
            gpas = gpas_soma * (1 + k * dist)

            if gpas < 0:
                seg.g_pas = 0
                print("WARNING!!! 'gpas' is in negative! Corrected to zero.")
            else:
                seg.g_pas = gpas

        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.apic[d](0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron


    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.apic[d](0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_ssoma_rdend_spatial(stim, d=30, k=0.001, gpas_soma=0.0001, Ra=100., cm=1., dt=0.1):
    """
    This simulation protocol  assumes spatial change of the 'gpas' parameter

    :param stim:
    :param d:     distance from soma in um
    :param m:     rate of linear change in gpas density moving away from soma (opt)
    :param gpas_soma: the value of gpas in soma (opt)
    :param Ra:    Axial resistance (opt)
    :param cm:    Membrane conductance (opt)
    :param dt:    Time step (opt)
    :return:
    """

    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm  # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas_soma  # gpas is a parameter to infer

        for seg in sec:
            h('soma distance()')
            dist = (h.distance(seg.x))
            gpas = gpas_soma * (1 + k * dist)

            if gpas < 0:
                seg.g_pas = 0
                print("WARNING!!! 'gpas' is in negative! Corrected to zero.")
            else:
                seg.g_pas = gpas

        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron

    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(h.apic[d](0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def real_morphology_model_srsoma_rdend_spatial(stim, d=30, k=0.001, gpas_soma=0.0001, Ra=100., cm=1., dt=0.1):
    """
    This simulation protocol  assumes spatial change of the 'gpas' parameter

    :param stim:
    :param d:     distance from soma in um
    :param m:     rate of linear change in gpas density moving away from soma (opt)
    :param gpas_soma: the value of gpas in soma (opt)
    :param Ra:    Axial resistance (opt)
    :param cm:    Membrane conductance (opt)
    :param dt:    Time step (opt)
    :return:
    """

    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra  # Ra is a parameter to infer
        sec.cm = cm  # parameter optimisation algorithm found this
        sec.v = 0

        sec.insert('pas')
        sec.g_pas = gpas_soma  # gpas is a parameter to infer

        for seg in sec:
            h('soma distance()')
            dist = (h.distance(seg.x))
            gpas = gpas_soma * (1 + k * dist)

            if gpas < 0:
                seg.g_pas = 0
                print("WARNING!!! 'gpas' is in negative! Corrected to zero.")
            else:
                seg.g_pas = gpas

        sec.e_pas = 0

    # Print information
    # h.psection()

    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt

    # Stimulus
    h.tstop = len(stim) * dt
    h.load_file("vplay.hoc")
    vec = h.Vector(stim)
    istim = h.IClamp(h.soma(0.5))
    vec.play(istim._ref_amp, h.dt)
    istim.delay = 0  # Just for Neuron
    istim.dur = 1e9  # Just for Neuron

    # Run simulation ->
    # Set up recording Vectors
    vs_vec = h.Vector()  # Membrane potential vector for soma recording
    vd_vec = h.Vector()  # Membrane potential vector for dendrite recording
    t_vec = h.Vector()  # Time stamp vector
    vd_vec.record(h.apic[d](0.5)._ref_v)
    vs_vec.record(h.soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    # h.tstop = 1200  # Simulation end
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v_soma = vs_vec.to_python()
    v_dend = vd_vec.to_python()
    
    v_soma = np.array(v_soma)
    v_dend = np.array(v_dend)

    v = np.concatenate((v_soma, v_dend))
    return t, v


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    from module.noise import white

    stim = []
    tv = np.linspace(0, 500, 5001)

    # Ramp Stimulus
    # s = 0.
    # for i, t in enumerate(tv):
    #     if t < 50:
    #         stim.append(0.)
    #     elif t < 400:
    #         stim.append(s)
    #         s += 0.0001
    #     else:
    #         stim.append(0.)
    # print s

    # F = 1.
    # f = F*1e-3
    # #fa = np.linspace(0, 150e-3, 5001)
    #
    # for i,t in enumerate(tv):
    #     stim.append(0.2*np.sin(2*np.pi*f*t))

    for i, t in enumerate(tv):
        if t < 50:
            stim.append(0.)
        elif t<= 410:
            stim.append(0.01)
        else:
            stim.append(0.)

    plt.figure(figsize=(12,7))
    plt.title("Stimulus")
    plt.xlabel("Time [ms]")
    plt.ylabel("I [uA]")
    plt.plot(tv, stim)
    #plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/imp.png")
    plt.show()

    # np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/stim.txt", stim)

    print((len(stim)))

    # --- Load NEURON morphology
    h('load_file("/home/terbe/parameter-inference/exp/morphology_131117-C2.hoc")')
    # Set the appropriate "nseg"
    for sec in h.allsec():
        sec.Ra = 150
    h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1
    t, v = real_morphology_model_ssoma_rdend_spatial(stim=stim)
    # v = white(7., v)

    plt.figure(figsize=(12, 7))
    plt.title("Voltage response")
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mV]")
    plt.plot(t, v)
    # plt.savefig("/Users/Dani/TDK/parameter_estim/stim_protocol2/steps/400/resp.png")
    plt.show()

    print((len(t)))
    print((len(v)))
