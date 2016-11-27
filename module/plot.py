"""
This module is to plot the results
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
import numpy as np
import os


def plot3d(param1, param2, z, title='', path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(param2.values, param1.values)
    ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contour(x, y, z, zdir='z', offset=0, cmap=CM.coolwarm)
    cset = ax.contour(x, y, z, zdir='x', offset=param2.range_min, cmap=CM.coolwarm)
    cset = ax.contour(x, y, z, zdir='y', offset=param1.range_max, cmap=CM.coolwarm)
    ax.set_title(title)
    ax.set_xlabel(param2.name + ' ' + param2.unit)
    ax.set_ylabel(param1.name + ' ' + param1.unit)
    filename = path + title + '_' + param1.name + str(param1.resolution) + '_' + \
               param2.name + str(param2.resolution) + '_'
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))


def marginal_plot(resolution, values, prior, likelihood, posterior, name='', unit='', paramset_name=''):
    # Plot posterior
    plt.figure()
    plt.title(name + " posterior (g) and prior (b) distribution")
    plt.xlabel(name + ' ' + unit)
    plt.ylabel("probability")
    plt.plot(values, posterior, '#34A52F')
    plt.plot(values, prior, color='#2FA5A0')

    filename = "/Users/Dani/TDK/parameter_estim/exp/out2/" + \
               paramset_name + '-' + name + "-posterior_" + str(resolution) + "_"
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))
    print "Plot done! File path: " + filename

    # Plot likelihood
    plt.figure()
    plt.title(name + " likelihood (r) and prior (b) distribution")
    plt.xlabel(name + ' ' + unit)
    plt.ylabel("probability")
    plt.plot(values, likelihood, color='#A52F34')
    plt.plot(values, prior, color='#2FA5A0')

    filename = "/Users/Dani/TDK/parameter_estim/exp/out2/" + \
               paramset_name + '-' + name + "-likelihood_" + str(resolution) + "_"
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))
    print "Plot done! File path: " + filename