"""
This module is to feature the distribution
and compare the prior and posterior one.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# TODO OUT OF RANGE EXCEPTION handling in a better way
def interpolate(x, y):
    """
    Cubic interpolation for the given symmetric distribution and test the resolution

    :param x: x codomain value vector
    :param y: y domain value vector
    :return: matrix with two row (x,y) or str on which direction out of range
    """
    max_idx = np.argmax(y)

    # Analysis
    value = y[max_idx] / 2.5

    try:
        left_idx = (np.abs(y[:max_idx] - value)).argmin()
    except ValueError:
        return "left"
    try:
        right_idx = len(y[:max_idx]) + (np.abs(y[max_idx:] - value)).argmin()
    except ValueError:
        return "right"

    data_points = len(x[left_idx:right_idx])

    if data_points < 8 or len(x[left_idx:max_idx]) < 3 or len(x[max_idx:right_idx]) < 4:
        print "\nData points on the left side: " + str(len(x[left_idx:max_idx]))
        print "Data points on the right side: " + str(len(x[max_idx:right_idx]))
        print "Data points to measure sharpness: " + str(data_points)
        print "\nWARNING! This trace is too sharp for this sampling frequency!\n" \
              "Note that the interpolation may not work efficiently \n" \
              "if the resolution is high enough so  -data point-  >= 8\n"

    # Interpolate
    f = interp1d(x, y, kind='cubic')
    x = np.linspace(x[0], x[len(x) - 1], 1000)
    y = f(x)

    t = np.ndarray(shape=(2, len(x)))
    t[0] = x
    t[1] = y

    return t


def sharpness(x, y):
    """
    Feature the sharpness of the given symmetric trace

    :param x: sample vector
    :param y: (symmetric) distribution vector
    :return: a characteristic scalar to feature the sharpness of the give distribution
    """
    max_idx = np.argmax(y)

    res = np.linspace(0.5, 1., 50, dtype=float, endpoint=False)
    full_dev = 0.
    for i in res:
        value = y[max_idx] * i
        left_idx = (np.abs(y[:max_idx] - value)).argmin()
        right_idx = len(y[:max_idx]) + (np.abs(y[max_idx:] - value)).argmin()

        full_dev += np.abs(x[right_idx] - x[left_idx])

    return full_dev / 50


def kl_test(posterior, prior, step, eps=0.0001):
    """
    Kullback-Leiber test for (numerically continuous) probability distributions.

    :param posterior: Posterior distribution codomain vector
    :param prior: Prior distribution codomain vector
    :param eps: fuzz factor, below this we avoid division with small values
    :return: KL divergence of the two given distribution
    """

    # Clip the too low values
    i = 0  # count deleted indexes
    for idx, item in enumerate(prior):
        if item < eps:
            prior = np.delete(prior, idx - i, axis=None)
            posterior = np.delete(posterior, idx - i, axis=None)
            i += 1

    # KL-divergence
    kdl = 0
    for i, p in enumerate(posterior):
        kdl += p * np.log(p / prior[i])*step

    return kdl


def stat(param):
    """
    Create statistic for the inference

    :param param: RandomVariable type
    :return: feature tuple (sigma, diff, pdiff, sharper) for the inference or str if out of range
    """
    true_param = param.value

    # Fit gaussian to posterior [the fitting may yield negative sigma values!!! ABS!]
    def gauss(x, mean, sigma):
        return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    p_init = [param.mean, param.sigma]

    p_opt = [0,0]
    p_err = [0,0]
    try:
        p_opt, p_cov = curve_fit(gauss, param.values, param.posterior, p0=p_init)
        p_err = np.sqrt(np.diag(p_cov))
    except ValueError:  # ValueError: array must not contain infs or NaNs
        p_opt[1] = param.sigma
        p_err[1] = 0.

    # Do some other statistics
    # TODO true_idx = (np.abs(x - true_param)).argmin() TypeError: unsupported operand type(s) for -: 'str' and 'float'
    if interpolate(param.values, param.posterior) is str:                      # In this case we cannot interpolate
        return interpolate(param.values, param.posterior)
    else:
        x = interpolate(param.values, param.posterior)[0]
        posterior = interpolate(param.values, param.posterior)[1]
        prior = interpolate(param.values, param.prior)[1]

        true_idx = (np.abs(x - true_param)).argmin()

        sharper = sharpness(x, prior) / sharpness(x, posterior)
        diff = np.abs(x[np.argmax(posterior)] - x[true_idx])
        accuracy = np.multiply(posterior[true_idx] / np.amax(posterior),100)

        return abs(p_opt[1]), diff, accuracy, sharper, p_err[1]


def re_sampling(old_res_trace, new_res):
    """
    Resampling trace

    :param old_res_trace: (elements number, 2) dimension np.ndarray()
    :param new_res: new domain elements array
    :return: The interpolated new resolution (new elements number, 2) dimension np.ndarray()
    """

    new_res_trace = np.ndarray((len(new_res), 2))
    f = interp1d(old_res_trace[:, 0], old_res_trace[:, 1])

    new_res_trace[:, 0] = new_res
    new_res_trace[:, 1] = f(new_res)

    return new_res_trace

if __name__ == "__main__":
    from matplotlib import pyplot
    import prior

    sigma = 5

    x = np.linspace(-50, 50, num=80)
    y = prior.normal(1, sigma, x)

    y_prior = prior.normal(0, 7, x)

    x_true = 0.

    diff, pdiff, sharper = stat(y, y_prior, x, x_true)
    print "Difference: " + str(diff)
    print "Probability mistake: " + str(pdiff)
    print "Sharper: " + str(sharper)

    pyplot.plot(x, y)
    pyplot.plot(x, y_prior)
    pyplot.show()
