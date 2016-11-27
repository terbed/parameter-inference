import numpy as np
from math import sqrt, log, cos, pi, exp


def white(sigma, v_vec, mu=0):
    """Returns the given array with white noise as numpy array"""
    noise_signal = np.random.normal(mu, sigma, len(v_vec))
    exp_v = np.add(v_vec, noise_signal)

    return np.array(exp_v)


def colored(D, lamb, dt, v_vec):
    """Returns the given array with colored noise as numpy array"""
    noise = []
    n, m = np.random.uniform(0.0, 1.0, 2)
    E = exp(-lamb * dt)
    e_0 = sqrt(-2 * D * lamb * log(m)) * cos(2 * pi * n)
    noise.append(e_0)

    for i in range(len(v_vec) - 1):
        a, b = np.random.uniform(0.0, 1.0, 2)
        h = sqrt(-2 * D * lamb * (1 - E ** 2) * log(a)) * cos(2 * pi * b)
        e_next = e_0 * E + h
        noise.append(e_next)
        e_0 = e_next

    return np.add(v_vec, noise)


# Solve the problem with generators and list comprehension ----------------------------------------------------------

def colored_noise_generator(D, lamb, dt):
    """
    An iterable generator function for colored noise.

    :param D: Amplitude of the noise
    :param lamb: Reciprocal of the characteristic time
    :param dt: Time step
    :return : yields the successive value
    """

    e_0 = None
    E = exp(-lamb * dt)

    while True:
        if e_0 is None:
            # Create the first value
            n, m = np.random.uniform(0.0, 1.0, 2)
            e_0 = sqrt(-2 * D * lamb * log(m)) * cos(2 * pi * n)
            yield e_0
        else:
            # Create succession
            a, b = np.random.uniform(0.0, 1.0, 2)
            h = sqrt(-2 * D * lamb * (1 - E ** 2) * log(a)) * cos(2 * pi * b)
            e_next = e_0 * E + h
            e_0 = e_next
            yield e_0


def colored_vector(D, lamb, dt, vec):
    """
    Ads colored noise to the given vector.
    This function uses the colored_noise_generator() generator function

    :param D: amplitude of the noise
    :param lamb: reciprocal of the characteristic time
    :param dt: time step
    :param vec: the vector to extend with noise
    :return: the given vector with noise
    """

    # Create color generator
    noise_generator = colored_noise_generator(D, lamb, dt)

    # iterate through vec and add noise then return the list
    return [x + noise_generator.next() for x in vec]
