import numpy as np
from numpy.linalg import inv


def generate(D, lamb, t):
    """Returns the covariance matrix defined by the parameters of the colored noise

    D: amplitude
    lamb: characteristic time constant (1/tau)
    t: the time sample values of the experiment
    """
    covmat = np.zeros((len(t), len(t)))

    for i, t1 in enumerate(t):
        for j, t2 in enumerate(t):
            covmat[i][j] = D * lamb * np.exp(-lamb * abs(t1 - t2))

    inv_covmat = inv(covmat)

    return np.array(inv_covmat)


if __name__ == "__main__":
    import sys
    import pandas as pd

    D = float(sys.argv[1])
    lamb = float(sys.argv[2])
    t_dur = float(sys.argv[3])
    length = float(sys.argv[4])
    dt = float(t_dur/length)

    t = np.linspace(0, t_dur, length)

    inv_covmat = generate(D, lamb, t)
    invcovmat = pd.DataFrame(data=inv_covmat.astype(float))
    invcovmat.to_csv('inv_covmat' + str(dt) + '.csv', header=False, float_format=None, index=False)

    print "DONE"

