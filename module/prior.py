import numpy as np
from functools import partial
from numpy import gradient


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


def get_mu(mean, var):
    """
    :param mean: Normal distribution mean 
    :param var: Normal distribution variance
    :return: Lognormal distribution mu
    """
    return np.log(mean/(1+var/mean**2))


def get_sig(mean, var):
    """
    :param mean: Normal distribution mean 
    :param var: Normal distribution variance
    :return: Lognormal distribution sigma
    """
    return np.sqrt(np.log(1+var/mean**2))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm as CM
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from module.probability import RandomVariable

    cm = RandomVariable("cm", range_min=0.99999, range_max=1.00001, resolution=10, mean=1., sigma=0.2)
    gpas = RandomVariable("gpas", range_min=0.00005, range_max=0.00015, resolution=1000, mean=0.0001, sigma=0.00002)

    d_step = cm.sigma*1e-6
    d_range = [cm.mean,]
    n = 2

    for i in range(n):
        d_range.append(d_range[0]+(i+1)*d_step)
        d_range.append(d_range[0]-(i+1)*d_step)

    normal = normal(d_range, cm.mean, cm.sigma)
    n = np.array(normal)

    def hessian(x, step):
        """
        Calculate the hessian matrix with finite differences
        Parameters:
           - x : ndarray
        Returns:
           an array of shape (x.dim, x.ndim) + x.shape
           where the array[i, j, ...] corresponds to the second derivative x_ij
        """
        print x.shape
        x_grad = np.gradient(x, step)
        hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
        for k, grad_k in enumerate(x_grad):
            # iterate over dimensions
            # apply gradient again to every component of the first derivative.
            tmp_grad = np.gradient(grad_k, step)
            for l, grad_kl in enumerate(tmp_grad):
                hessian[k, l, :, :] = grad_kl
        return hessian

    plt.figure(figsize=(12,7))
    plt.plot(d_range, n, 'o')
    plt.show()
