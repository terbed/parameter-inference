import numpy as np
from functools import partial


def normal_val(x, mean, sigma):
    """Returns a value for the given x"""
    normal_distribution_value = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
    return normal_distribution_value


def normal(vec, mean, sigma):
    """Returns a normal distribution as a numpy array
        equal to map(partial(normal_val, mean=mean, sigma=sigma), vec)
    """

    return [partial(normal_val, mean=mean, sigma=sigma)(x) for x in vec]


def normal2d(x_mean, x_sigma, x_values, y_mean, y_sigma, y_values):
    """Returns a two dimensional normal distribution as a numpy array"""
    prior = np.zeros((len(x_values), len(y_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            prior[i, j] = normal_val(x_mean, x_sigma, x) * normal_val(y_mean, y_sigma, y)

    return prior


def normal_nd(*priors):
    """
    General normal distribution

    :param priors: RandomVariable objects
    :return: n dimensional normal distribution
    """
    # Trivial case
    if len(priors) == 1:
        return priors

    # General case
    shape = []
    for item in priors:
        shape.append(len(item.values))

    n = np.ones(shape)
    for idx, _ in np.ndenumerate(n):
        for ax, element in enumerate(idx):
            n[idx] *= priors[ax].values[element]

    return n


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm as CM
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    x1 = np.linspace(-100, 100, 20)
    y1 = normal(0, 18, x1)

    x2 = np.linspace(-50, 50, 30)
    y2 = normal(0, 3, x2)

    x3 = np.linspace(-30, 30, 40)
    y3 = normal(0, 6, x3)

    x4 = np.linspace(-10, 10, 50)
    y4 = normal(0, 2, x4)

    y1y2y3 = normal_nd(y1, y2, y3, y4)

    y11 = y1y2y3.sum(axis=(1, 2, 3)) * abs(x2[1]-x2[0]) * abs(x3[1]-x3[0]) * abs(x4[1]-x4[0])

    plt.figure()
    plt.plot(x1, y1, 'r')
    plt.plot(x1, y11)
    plt.show()

    y1y2 = np.sum(y1y2y3, axis=(2, 3))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(x2, x1)
    ax.plot_surface(x, y, y1y2, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contour(x, y, y1y2, zdir='z', offset=-0, cmap=CM.coolwarm)
    cset = ax.contour(x, y, y1y2, zdir='x', offset=0.00004, cmap=CM.coolwarm)
    cset = ax.contour(x, y, y1y2, zdir='y', offset=160, cmap=CM.coolwarm)
    ax.set_title('Posterior')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()