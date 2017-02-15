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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm as CM
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from module.probability import RandomVariable

    cm = RandomVariable("cm", range_min=0.5, range_max=1.5, resolution=80, mean=1., sigma=0.05)
    gpas = RandomVariable("gpas", range_min=0.00005, range_max=0.00015, resolution=80, mean=0.0001, sigma=0.00002)

    prior = normal_nd(cm, gpas)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(gpas.values, cm.values)
    ax.plot_surface(x, y, prior, rstride=1, cstride=1, alpha=0.3, cmap=CM.rainbow)
    cset = ax.contourf(x, y, prior, zdir='z', offset=-0, cmap=CM.rainbow)
    cset = ax.contourf(x, y, prior, zdir='x', offset=gpas.range_min, cmap=CM.rainbow)
    cset = ax.contourf(x, y, prior, zdir='y', offset=cm.range_max, cmap=CM.rainbow)
    ax.set_title('Posterior')
    ax.set_xlabel('x2')
    ax.set_ylabel('x3')
    plt.show()