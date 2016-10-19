"""
This module is to feature the sharpness of the trace.
It works well with gauss like distributions (quadratic interpolation)
And with huge enough resolution
"""

import numpy as np
from scipy.interpolate import interp1d


def sharpness(x, y):
    # The x value for the maximum y:
    max_x = np.argmax(y)

    # Analysis ---
    value = y[max_x]/2.

    left_idx = (np.abs(y[:max_x] - value)).argmin()
    right_idx = len(y[:max_x]) + (np.abs(y[max_x:] - value)).argmin()

    data_points = len(x[left_idx:right_idx]) + 1

    if data_points < 8 or len(x[left_idx:max_x]) < 3 or len(x[max_x:right_idx]) < 4:
        print "\nData points on the left side: " + str(len(x[left_idx:max_x]) + 1)
        print "Data points on the right side: " + str(len(x[max_x:right_idx]))
        print "Data points to measure sharpness: " + str(data_points)
        print "\nWARNING! This trace is too sharp for this sampling frequency!\n" \
              "Note that the sharpness value is exactly characteristic for the given trace\n" \
              "if the resolution is high enough so  -data point-  >= 8"

    # Sharpness ---
    f = interp1d(x, y, kind='cubic')
    x_new = np.linspace(x[0], x[len(x) - 1], 1000)
    y_new = f(x_new)
    max_x = np.argmax(y_new)

    res = np.linspace(1.1, 2., 50, dtype=float)
    full_dev = 0.
    for i in res:
        value = y_new[max_x] / i
        left_idx = (np.abs(y_new[:max_x] - value)).argmin()
        right_idx = len(y_new[:max_x]) + (np.abs(y_new[max_x:] - value)).argmin()

        full_dev += np.abs(x_new[left_idx] - x_new[right_idx])

    return full_dev/50


if __name__ == "__main__":
    from matplotlib import pyplot
    import prior

    sigma = 5

    x = np.linspace(-50, 50, num=50)
    y = prior.normal(0, sigma, x)

    x_exact = np.linspace(-50, 50, num=1000)  # High enough resolution for reliable sharpness check
    y_exact = prior.normal(0, sigma, x_exact)

    pyplot.figure()
    pyplot.title(" Low res trace: b, High res trace: r")
    pyplot.plot(x, y, 'bo')
    pyplot.plot(x_exact, y_exact, 'r-')
    pyplot.xlabel("x")
    pyplot.ylabel("y")

    print "\nSharpness of the given trace: " + str(sharpness(x, y))
    print "Sharpness of the high res trace: " + str(sharpness(x_exact, y_exact))

    pyplot.show()
