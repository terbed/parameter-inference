import numpy as np
from numpy.linalg import inv
from functools import partial


def E(dt, D, lamb):
    return D * lamb * np.exp(-lamb * dt)


def generate(D, lamb, t):
    """Returns the covariance matrix defined by the parameters of the colored noise

    D: amplitude
    lamb: characteristic time constant (1/tau)
    t: the time sample values of the experiment
    """
    Exp = partial(E, D=D, lamb=lamb)
    covmat = [[Exp(abs(t1 - t2)) for t2 in t] for t1 in t]
    inv_covmat = inv(covmat)

    return np.array(inv_covmat)


if __name__ == "__main__":
    import pandas as pd

    D = 30
    lamb = 0.1
    t_dur = 200
    dt = 0.1

    t = np.linspace(0, t_dur, t_dur/dt)

    inv_covmat = generate(D, lamb, t)
    invcovmat = pd.DataFrame(data=inv_covmat.astype(float))
    invcovmat.to_csv('inv_covmatika' + str(dt) + '.csv', header=False, float_format=None, index=False)

    print "DONE"

