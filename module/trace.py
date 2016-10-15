import numpy as np


def sharpness(x, y):
    res = np.linspace(1.1, 2., 100, dtype=float)
    max_x = np.argmax(y)

    full_dev = 0.
    for i in res:
        value = y[max_x] / i
        left_idx = (np.abs(y[:max_x] - value)).argmin()
        right_idx = len(y[:max_x]) + (np.abs(y[max_x:] - value)).argmin()

        full_dev += np.abs(x[left_idx] - x[right_idx])

    return full_dev


if __name__ == "__main__":
    from matplotlib import pyplot
    import prior
    x = np.linspace(-50, 50, num=100)

    y1 = prior.normal(0, 10, x)
    y2 = prior.normal(0, 3, x)
    y3 = prior.normal(0, 2, x)
    y4 = prior.normal(0, 1, x)

    pyplot.figure()
    pyplot.title(" Test r:10, b:3, g:2, y:1 ")
    pyplot.plot(x, y1, 'r')
    pyplot.plot(x, y2, 'b')
    pyplot.plot(x, y3, 'g')
    pyplot.plot(x, y4, 'y')
    pyplot.xlabel("x")
    pyplot.ylabel("y")

    print "Sharpness of the red gauss: " + str(sharpness(x, y1))
    print "Sharpness of the blue gauss: " + str(sharpness(x, y2))
    print "Sharpness of the green gauss: " + str(sharpness(x, y3))
    print "Sharpness of the yellow gauss, which is a standard normal distribution: " + str(sharpness(x, y4))
    pyplot.show()

