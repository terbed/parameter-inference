"""
This module is to feature the distribution
and compare the prior and posterior one.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from module.prior import normal_val, normal


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

    res = np.linspace(0.5, 1., 50, endpoint=False, dtype=float)
    full_dev = 0.
    for i in res:
        value = y[max_idx] * i

        try:
            left_idx = (np.abs(y[:max_idx] - value)).argmin()
            right_idx = len(y[:max_idx]) + (np.abs(y[max_idx:] - value)).argmin()
            full_dev += np.abs(x[right_idx] - x[left_idx])
        except ValueError:
            print "ValueError in sharpness checking!"
            from matplotlib import pyplot as plt
            plt.figure()
            plt.title("Fitted posterior")
            plt.plot(x,y)
            plt.show()

    return full_dev / 50.


def fit_normal(x, y, *p_init):
    try:
        p_opt, p_cov = curve_fit(normal_val, x, y, p0=p_init)
        p_err = np.sqrt(np.diag(p_cov))
        return (p_opt, p_err)
    except (ValueError, RuntimeError) as err:
        # ValueError: array must not contain infs or NaNs
        # RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 600.
        print "Something went wrong fitting gauss to data...\n"
        print(err)
        print(err.args)
        return ([None, None], [None, None])


def analyse(param, p_opt):
    """
    This function analyses the marginal posterior for the given parameter
    :param param: RandomVariable object after inference
    :param p_opt: The initial values for normal distribution fitting to posterior data
    :return: (fitted_sigma, fit_err, relative_deviation, acc, sharper, broader) tuple
    """

    # Create high resolution (prior and) posterior from fitted function
    x = np.linspace(param.range_min, param.range_max, 3000)
    prior = normal(x, param.mean, param.sigma)
    posterior = normal(x, p_opt[0][0], p_opt[0][1])
    # Ensure that posterior is in range: -> Wrong solutions! It changes the shaprness value!!!
    # xp = np.linspace(p_opt[0][0] - 2*p_opt[0][1], p_opt[0][0] + 2*p_opt[0][1], 3000)

    # from matplotlib import pyplot as plt
    # plt.close()
    # plt.title(param.name)
    # plt.plot(x, prior, label="prior: %.3e" % sharpness(x,prior))
    # plt.plot(x, posterior, label="posterior: %.3e" % sharpness(x, posterior))
    # plt.plot(x, posterior)
    # plt.legend(loc="best")
    # plt.show()

    # Do some statistics
    true_idx = (np.abs(x - param.value)).argmin()

    sharper = sharpness(x, prior) / sharpness(x, posterior)
    broadness = sharpness(x, posterior) / sharpness(x, prior) * 100
    rdiff = (param.value - p_opt[0][0]) / param.value * 100
    accuracy = posterior[true_idx] / np.amax(posterior) * 100

    # The relative sigma + mean error
    fit_err = (abs(p_opt[1][1])/abs(p_opt[0][1]) + abs(p_opt[1][0])/abs(p_opt[0][0]))*100

    return abs(p_opt[0][1]), fit_err, rdiff, accuracy, sharper, broadness


# OLD...
def stat(param):
    """
    Create statistic for the inference

    :param param: RandomVariable type
    :return: feature tuple (sigma, diff, pdiff, sharper, sigma_err, rdiff) for the inference or str if out of range
    """
    true_param = param.value

    p_init = [param.mean, param.sigma]
    p_opt = [0,0]
    p_err = [0,0]
    try:
        p_opt, p_cov = curve_fit(normal_val, param.values, param.posterior, p0=p_init)
        p_err = np.sqrt(np.diag(p_cov))
        print "\nError of fitting gauss to posterior: " + str(p_err)
    except (ValueError, RuntimeError) as err:
        # ValueError: array must not contain infs or NaNs
        # RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 600.
        print "Something went wrong fitting gauss to data...\n"
        print(err)
        print(err.args)
        p_opt[1] = param.sigma
        p_err[1] = 0.
        return abs(p_opt[1]), param.sigma, 0., 1., p_err[1], 0.

    # Create high resolution (prior and) posterior from fitted function
    x = np.linspace(param.range_min, param.range_max, 10000)
    prior = normal(x, param.mean, param.sigma)
    posterior = normal(x, p_opt[0], p_opt[1])

    # Range for posterior: [mean-2sig, mean+2sig]
    x2 = np.linspace(p_opt[0]-2*p_opt[1], p_opt[0]+2*p_opt[1], 10000)
    posharp = normal(x2, p_opt[0], p_opt[1])

    # Do some statistics
    true_idx = (np.abs(x - true_param)).argmin()

    sharper = sharpness(x, prior) / sharpness(x2, posharp)
    broader = sharpness(x2, posharp) / sharpness(x, prior)
    diff = p_opt[0] - true_param
    accuracy = np.multiply(posterior[true_idx] / np.amax(posharp), 100)

    return abs(p_opt[1]), diff, accuracy, sharper, p_err[1], broader


def kl_test(posterior, prior, step, eps=0.000001):
    """
    Kullback-Leiber test for (numerically continuous) probability distributions.

    :param posterior: Posterior distribution codomain vector
    :param prior: Prior distribution codomain vector
    :parameter step: parameter step value
    :param eps: fuzz factor, below this we avoid division with small values
    :return: KL divergence of the two given distribution
    """

    pi = []
    post = []

    # Clip the too low values
    for idx, item in enumerate(prior):
        if item > eps and posterior[idx] > eps:
            pi.append(item)
            post.append(posterior[idx])


    # KL-divergence
    kdl = 0.
    for i, p in enumerate(post):
        kdl += p * np.log(p / pi[i])*step

    return kdl


def re_sampling(old_res_trace, new_res):
    """
    Resampling trace

    :param old_res_trace: np.ndarray(old_len, 2) o_trace[:,0] = ot_vec; o_trace[:, 1] = ov_vec;
    :param new_res: new domain elements array (time vector)
    :return: The interpolated new resolution (new elements number, 2) dimension np.ndarray()
    """

    new_res_trace = np.ndarray((len(new_res), 2))
    f = interp1d(old_res_trace[:,0], old_res_trace[:,1])

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
