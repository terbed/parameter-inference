from neuron import h, gui


def one_compartment(cm=1., gpas=0.0001, dt=0.1, stype='both'):
    """ One compartment simulation, variables: membrane capacitance and passive conductance """
    # Creating one compartment passive  model (interacting with neuron)
    soma = h.Section(name='soma')
    soma.L = soma.diam = 50                             # it is a sphere
    soma.v = -70
    soma.cm = cm                                        # parameter to infer

    # Insert passive conductance
    soma.insert('pas')
    soma.g_pas = gpas                                  # parameter to infer
    soma.e_pas = -70

    # Creating stimulus
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
    else:
        h.tstop = 400
        stim1 = h.IClamp(soma(0.5))
        stim1.delay = 10
        stim1.amp = 0.5
        stim1.dur = 5

        stim2 = h.IClamp(soma(0.5))
        stim2.delay = 120
        stim2.amp = 0.1
        stim2.dur = 175

    # Print Information
    # h.psection()

    # Run simulation ->
    # Set up recording Vectors
    v_vec = h.Vector()                                  # Membrane potential vector
    t_vec = h.Vector()                                  # Time stamp vector
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.dt = dt                                           # Time step (iteration)
    h.steps_per_ms = 1 / dt

    h.v_init = -70
    h.finitialize(h.v_init)                             # Starting membrane potential

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def stick_and_ball(Ra=100, gpas=0.0001, cm=1., Ra_max=250., dt=0.1, stype='broad'):
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

    # Set the appropriate "nseg"
    for sec in h.allsec():
        sec.Ra = Ra_max
    h('forall {nseg = int((L/(0.1*lambda_f(100))+.9)/2)*2 + 1}')  # If Ra_max = 105 dend.nseg = 21 and soma.nseg = 1

    # -- Biophysics --
    # Sec parameters and conductance
    for sec in h.allsec():
        sec.Ra = Ra                                                 # Ra is a parameter to infer
        sec.cm = cm
        sec.v = -70

        sec.insert('pas')
        sec.g_pas = gpas                                            # gpas is a parameter to infer
        sec.e_pas = -70

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
    else:
        h.tstop = 400
        stim1 = h.IClamp(soma(0.5))
        stim1.delay = 10
        stim1.amp = 0.5
        stim1.dur = 5

        stim2 = h.IClamp(soma(0.5))
        stim2.delay = 120
        stim2.amp = 0.1
        stim2.dur = 175

    # Run simulation ->
    # Print information
    # h.psection()

    # Set up recording Vectors
    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.dt = dt                                                       # Time step (iteration)
    h.steps_per_ms = 1 / dt

    h.v_init = -70
    h.finitialize(h.v_init)                                         # Starting membrane potential

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


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
    h.tstop = 1200  # Simulation end
    h.dt = dt  # Time step (iteration)
    h.steps_per_ms = 1 / dt
    h.v_init = 0
    h.finitialize(h.v_init)

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v