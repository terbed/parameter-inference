import numpy as np
from functools import partial


def normal_val(x, mean, sigma):
    """Returns a value for the given x"""
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def normal(vec, mean, sigma):
    """Returns a normal distribution as a numpy array
        equal to map(partial(normal_val, mean=mean, sigma=sigma), vec)
    """
    normal_val_map = partial(normal_val, mean=mean, sigma=sigma)
    return [normal_val_map(x) for x in vec]


def normal2d(x_mean, x_sigma, x_values, y_mean, y_sigma, y_values):
    """Returns a two dimensional normal distribution as a numpy array

    The solution is equivalent to:
        prior = np.zeros((len(x_values), len(y_values)))
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                prior[i, j] = normal_val(x_mean, x_sigma, x) * normal_val(y_mean, y_sigma, y)
        return prior
    """

    normal_val_xmap = partial(normal_val, mean=x_mean, sigma=x_sigma)
    normal_val_ymap = partial(normal_val, mean=y_mean, sigma=y_sigma)
    return [[normal_val_xmap(x) * normal_val_ymap(y) for y in y_values] for x in x_values]


def normal_nd(*params):
    """
    General normal distribution

    :param params: RandomVariable objects
    :return: n dimensional normal distribution
    """
    # Trivial case
    if len(params) == 1:
        return params[0].prior

    # General case
    shape = []
    for item in params:
        shape.append(len(item.values))

    n = np.ones(shape)
    for idx, _ in np.ndenumerate(n):
        for ax, element in enumerate(idx):
            n[idx] *= params[ax].prior[element]

    return n


def log_normal_val(x, mu, sigma):
    return 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))


def log_normal(vec, mu, sigma):
    lognormal = partial(log_normal_val, mu=mu, sigma=sigma)
    return [lognormal(x) for x in vec]

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm as CM
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from module.probability import RandomVariable

    cm = RandomVariable("cm", range_min=50, range_max=150, resolution=1000, mean=100., sigma=20.)
    gpas = RandomVariable("gpas", range_min=0.00005, range_max=0.00015, resolution=1000, mean=0.0001, sigma=0.00002)

    def get_mu(mean, var):
        return np.log(mean/(1+var/mean**2))

    def get_sig(mean, var):
        return np.sqrt(np.log(1+var/mean**2))

    normal = normal(cm.values, cm.mean, cm.sigma)
    mu = get_mu(cm.mean, cm.sigma**2)
    sig = get_sig(cm.mean, cm.sigma**2)
    print mu
    print sig

    lognormal = log_normal(cm.values, mu, sig-0.15)

    plt.figure(figsize=(12,7))
    plt.plot(cm.values, normal)
    plt.plot(cm.values, lognormal)
    plt.show()