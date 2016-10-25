from neuron import h, gui


def one_compartment(cm, g_pas, dt=0.1):
    """ One compartment simulation, variables: membrane capacitance and passive conductance """
    # Creating one compartment passive  model (interacting with neuron)
    soma = h.Section(name='soma')
    soma.L = soma.diam = 50                             # it is a sphere
    soma.v = -70
    soma.cm = cm                                        # parameter to infer

    # Insert passive conductance
    soma.insert('pas')
    soma.g_pas = g_pas                                  # parameter to infer
    soma.e_pas = -70

    # Creating stimulus
    stim = h.IClamp(soma(0.5))
    stim.delay = 30
    stim.amp = 0.1
    stim.dur = 100

    # Print Information
    # h.psection()

    # Set up recording Vectors
    v_vec = h.Vector()                                  # Membrane potential vector
    t_vec = h.Vector()                                  # Time stamp vector
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.tstop = 200                                       # Simulation end
    h.dt = dt                                           # Time step (iteration)
    h.steps_per_ms = 1 / dt

    h.v_init = -70
    h.finitialize(h.v_init)                             # Starting membrane potential

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v


def stick_and_ball(Ra, gpas, Ra_max=150., dt=0.1):
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
        sec.cm = 1
        sec.v = -70

        sec.insert('pas')
        sec.g_pas = gpas                                            # gpas is a parameter to infer
        sec.e_pas = -70

    # Stimulus
    stim = h.IClamp(soma(0.5))
    stim.delay = 30
    stim.amp = 0.1
    stim.dur = 100

    # Print information
    # h.psection()

    # Set up recording Vectors
    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    # Simulation duration and RUN
    h.tstop = 200                                                   # Simulation end
    h.dt = dt                                                       # Time step (iteration)
    h.steps_per_ms = 1 / dt

    h.v_init = -70
    h.finitialize(h.v_init)                                         # Starting membrane potential

    h.init()
    h.run()

    t = t_vec.to_python()
    v = v_vec.to_python()

    return t, v
