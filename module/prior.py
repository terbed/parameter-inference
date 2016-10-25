import numpy as np


def normal_val(mean, sigma, x):
    """Returns a value for the given x"""
    normal_distribution_value = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
    return normal_distribution_value


def normal(mean, sigma, values):
    """Returns a normal distribution as a numpy array"""
    prior = []
    for x in values:
        prior.append(normal_val(mean, sigma, x))

    return prior


def normal2d(x_mean, x_sigma, x_values, y_mean, y_sigma, y_values):
    """Returns a two dimensional normal distribution as a numpy array"""
    prior = np.zeros((len(x_values), len(y_values)))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            prior[i, j] = normal_val(x_mean, x_sigma, x) * normal_val(y_mean, y_sigma, y)

    return prior


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = np.linspace(-50, 50, 1000)
    y = normal(0, 6, x)

    plt.figure()
    plt.plot(x, y)
    plt.show()
    print np.sum(y)*np.abs(x[0]-x[1])

    x1 = np.linspace(-50, 50, 1000)
    x2 = np.linspace(-50, 50, 1000)

    y = normal2d(0, 5, x1, 2, 6, x2)
    print np.sum(y)*np.abs(x1[1]-x1[0])*np.abs(x2[1]-x2[0])

    plt.figure()
    plt.plot(x1, np.sum(y, axis=1), 'r')
    plt.plot(x2, np.sum(y, axis=0), "b")
    plt.show()
