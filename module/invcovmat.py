import numpy as np
from numpy.linalg import inv
from functools import partial


def E(dt, D, lamb):
    return D * lamb * np.exp(-lamb * dt)


def generate(D, lamb, t):
    """Returns the covariance matrix defined by the parameters of the colored noise

    :param D: amplitude
    :param lamb: characteristic time constant (1/tau)
    :param t: the time sample values of the experiment
    """
    Exp = partial(E, D=D, lamb=lamb)
    covmat = [[Exp(abs(t1 - t2)) for t2 in t] for t1 in t]
    inv_covmat = inv(covmat)

    return np.array(inv_covmat)


# Diagonal matrix for with noise..
def diagonal(sigma, len):
    d = 1/(sigma**2)

    inv_covmat = np.zeros((len, len), dtype=float)

    for i in range(len):
        inv_covmat[i, i] = d

    return inv_covmat


if __name__ == "__main__":
    import pandas as pd

    # D = 30
    # lamb = 0.1
    # t_dur = 200
    # dt = 0.1
    #
    # t = np.linspace(0, t_dur, t_dur/dt)
    #

    inv_covmat = diagonal(2., 12001)
    print inv_covmat.shape
    invcovmat = pd.DataFrame(data=inv_covmat.astype(float))
    invcovmat.to_csv('inv_covmat_w.csv', header=False, float_format=None, index=False)

    #
    # print "DONE"

