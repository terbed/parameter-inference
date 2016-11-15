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
