"""
This module is to plot the results
"""
from module.prior import normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as CM
import matplotlib.patches as mpatches
import os
import numpy as np
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from module.prior import normal


def check_directory(working_path):
    if not os.path.exists(working_path):
        os.makedirs(working_path)


def save_file(X, path, name, header=''):
    i = 0
    while os.path.exists('{}({:d}).txt'.format(path + "/" + name, i)):
        i += 1
    np.savetxt('{}({:d}).txt'.format(path + "/" + name, i), X, header=header, delimiter='\t')


def save_params(params, path):
    for item in params:
        i = 0
        while os.path.exists('{}({:d}).txt'.format(path + "/" + item.name, i)):
            i += 1
        np.savetxt('{}({:d}).txt'.format(path + "/" + item.name, i), item.init, fmt="%s")

def fullplot(result):
    """

    :param result: Inference object after computed results
    :return: Plots a joint and marginal full plot
    """
    from module.prior import normal

    plt.close('all')
    pnum = len(result.p.params)

    # Check there is more parameters
    if pnum < 2:
        return 0

    num_of_plot = pnum + (pnum ** 2 - pnum) / 2

    f, ax = plt.subplots(pnum, pnum, figsize=(14, 9))
    f.subplots_adjust(hspace=.001, wspace=.001)

    for row in range(pnum):
        for col in range(pnum):
            # Marginal plots
            if row == col:
                i = row
                ax[i, i].grid()
                ax[row, col].set_xlabel(result.p.params[i].name + ' ' + result.p.params[i].unit)
                ax[row, col].plot(result.p.params[i].values, result.p.params[i].likelihood, marker='o', color="#ffc82e")
                ax[row, col].axvline(result.p.params[i].value, color='#3acead', linewidth=0.8, alpha=1)
                ax[row, col].axvline(result.p.params[i].max_l, color='#FF1493', linewidth=0.8, alpha=1, linestyle='dashed')

                ax[row, col].tick_params(axis='y', which='both', left='off', right='on', labelleft='off',
                                         labelright='on')
                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off',
                                         labeltop='off')
                ax[row, col].xaxis.set_label_position('top')

                if row == pnum - 1:
                    ax[row, col].tick_params(axis='x', which='both', top='off', bottom='on', labelbottom='on',
                                             labeltop='off')
            # Joint plots
            elif col < row:
                likelihood = result.likelihood
                ax[row, col].grid()
                ax[row, col].axvline(result.p.params[col].value, color='#3acead', linewidth=0.8, alpha=1)
                ax[row, col].axvline(result.p.params[col].max_l, color='#FF1493', linewidth=0.8, alpha=1, linestyle='dashed')
                ax[row, col].axhline(result.p.params[row].value, color='#3acead', linewidth=0.8, alpha=1)
                ax[row, col].axhline(result.p.params[row].max_l, color='#FF1493', linewidth=0.8, alpha=1, linestyle='dashed')

                # Marginalize if needed
                if len(result.p.params) > 2:
                    for idx, item in enumerate(result.p.params):
                        if idx != row and idx != col:
                            likelihood = np.sum(likelihood, axis=idx) * item.step

                # Contour plot
                y, x = np.meshgrid(result.p.params[row].values, result.p.params[col].values)
                cs = ax[row, col].contour(x, y, likelihood, cmap=CM.rainbow)
                if col == 0:
                    ax[row, col].set_ylabel(result.p.params[row].name + ' ' + result.p.params[row].unit)
                # ax[row, col].clabel(cs, inline=1, fontsize=5)

                # Set up labels
                ax[row, col].tick_params(axis='y', which='both', left='off', right='off', labelleft='off',
                                         labelright='off')
                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off',
                                         labeltop='off')

                if row == pnum - 1:
                    ax[row, col].tick_params(axis='x', which='both', top='off', bottom='on', labelbottom='on',
                                             labeltop='off')
                    ax[row, col].set_xlabel(result.p.params[col].name + ' ' + result.p.params[col].unit)

                if col == 0:
                    ax[row, col].tick_params(axis='y', which='both', left='on', right='off', labelleft='on',
                                             labelright='off')
            else:
                ax[row, col].set_axis_off()

    i = 0
    while os.path.exists('{}({:d}).png'.format(result.working_path + "/fullplot_L", i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(result.working_path + "/fullplot_L", i))

    f, ax = plt.subplots(pnum, pnum, figsize=(14, 9))
    f.subplots_adjust(hspace=.001, wspace=.001)

    for row in range(pnum):
        for col in range(pnum):
            # Marginal plots
            if row == col:
                i = row
                ax[i, i].grid()
                ax[row, col].axvline(result.p.params[i].value, color='#3acead', linewidth=0.8, alpha=1)
                ax[row, col].axvline(result.p.params[i].max_p, color='#FF1493', linewidth=0.8, alpha=1, linestyle='dashed')

                ax[row, col].set_xlabel(result.p.params[i].name + ' ' + result.p.params[i].unit)
                ax[row, col].plot(result.p.params[i].values, result.p.params[i].posterior, 'o', color="#FF5F2E",
                                  label="posterior")

                tt = np.linspace(result.p.params[i].range_min, result.p.params[i].range_max, 2000)
                fitted = normal(tt, result.p.params[i].fitted_gauss[0][0], result.p.params[i].fitted_gauss[0][1])
                ax[row, col].plot(tt, fitted, color="#FF5F2E", label="fitted")

                ax[row, col].plot(result.p.params[i].values, result.p.params[i].prior, color="#2EFFC8", label="prior")
                # leg = ax[row, col].legend()
                # leg.get_frame().set_alpha(0.3)

                ax[row, col].tick_params(axis='y', which='both', left='off', right='on', labelleft='off',
                                         labelright='on')
                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off',
                                         labeltop='off')
                ax[row, col].xaxis.set_label_position('top')

                if row == pnum - 1:
                    ax[row, col].tick_params(axis='x', which='both', top='off', bottom='on', labelbottom='on',
                                             labeltop='off')
            # Joint plots
            elif col < row:
                posterior = result.posterior
                ax[row, col].grid()
                ax[row, col].axvline(result.p.params[col].value, color='#3acead', linewidth=0.8, alpha=1)
                ax[row, col].axvline(result.p.params[col].max_p, color='#FF1493', linewidth=0.8, alpha=1, linestyle='dashed')
                ax[row, col].axhline(result.p.params[row].value, color='#3acead', linewidth=0.8, alpha=1)
                ax[row, col].axhline(result.p.params[row].max_p, color='#FF1493', linewidth=0.8, alpha=1, linestyle='dashed')

                # Marginalize if needed
                if len(result.p.params) > 2:
                    for idx, item in enumerate(result.p.params):
                        if idx != row and idx != col:
                            posterior = np.sum(posterior, axis=idx) * item.step

                # Contour plot
                y, x = np.meshgrid(result.p.params[row].values, result.p.params[col].values)
                cs = ax[row, col].contour(x, y, posterior, cmap=CM.rainbow)
                if col == 0:
                    ax[row, col].set_ylabel(result.p.params[row].name + ' ' + result.p.params[row].unit)
                # ax[row, col].clabel(cs, inline=1, fontsize=5)

                # Set up labels
                ax[row, col].tick_params(axis='y', which='both', left='off', right='off', labelleft='off',
                                         labelright='off')
                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off',
                                         labeltop='off')

                if row == pnum - 1:
                    ax[row, col].tick_params(axis='x', which='both', top='off', bottom='on', labelbottom='on',
                                             labeltop='off')
                    ax[row, col].set_xlabel(result.p.params[col].name + ' ' + result.p.params[col].unit)

                if col == 0:
                    ax[row, col].tick_params(axis='y', which='both', left='on', right='off', labelleft='on',
                                             labelright='off')
            else:
                ax[row, col].set_axis_off()

    i = 0
    while os.path.exists('{}({:d}).png'.format(result.working_path + "/fullplot_P", i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(result.working_path + "/fullplot_P", i))


def plot_joint(result, param1, param2):
    """
    Plot 2 parameters with single marginal plots and a 3d plot

    :param result: Inference objekt
    :param param1 param2: RandomVariable Objekt
    """

    path = result.working_path + "/joint"
    check_directory(path)

    ax1 = 0
    ax2 = 0
    likelihood = result.likelihood
    posterior = result.posterior

    # Axes of parameters to be plotted
    for idx, item in enumerate(result.p.params):
        if param1.name == item.name:
            ax1 = idx
        if param2.name == item.name:
            ax2 = idx

    # Set up likelihood and posterior
    # Marginalize if more parameters...
    if len(result.p.params) > 2:
        for idx, item in enumerate(result.p.params):
            if idx != ax1 and idx != ax2:
                likelihood = np.sum(likelihood, axis=idx) * item.step
                posterior = np.sum(posterior, axis=idx) * item.step

    print "Is the JOINT posterior a probability distribution? Integrate(posterior) = " + str(
        np.sum(posterior) * param1.step * param2.step)

    # 3d plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(param2.values, param1.values)
    ax.plot_surface(x, y, likelihood, rstride=1, cstride=1, alpha=0.3, cmap=CM.rainbow)
    ax.contour(x, y, likelihood, zdir='z', offset=0, cmap=CM.rainbow)
    ax.contourf(x, y, likelihood, zdir='x', offset=param2.range_min, cmap=CM.rainbow)
    ax.contourf(x, y, likelihood, zdir='y', offset=param1.range_max, cmap=CM.rainbow)
    ax.set_title("Joint likelihood")
    ax.set_xlabel(param2.name + ' ' + param2.unit)
    ax.set_ylabel(param1.name + ' ' + param1.unit)
    filename = path + '/L_' + param1.name + '-' + param2.name
    i = 0
    while os.path.exists('{}({:d}).png'.format(filename, i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(filename, i))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(param2.values, param1.values)
    ax.plot_surface(x, y, posterior, rstride=1, cstride=1, alpha=0.3, cmap=CM.rainbow)
    # ax.plot_surface(x, y, result.p.joint_prior, rstride=1, cstride=1, alpha=0.3, cmap=CM.rainbow)
    ax.contour(x, y, posterior, zdir='z', offset=0, cmap=CM.rainbow)
    ax.contourf(x, y, posterior, zdir='x', offset=param2.range_min, cmap=CM.rainbow)
    ax.contourf(x, y, posterior, zdir='y', offset=param1.range_max, cmap=CM.rainbow)
    ax.set_title("Joint posterior")
    ax.set_xlabel(param2.name + ' ' + param2.unit)
    ax.set_ylabel(param1.name + ' ' + param1.unit)
    filename = path + '/P_' + param1.name + '-' + param2.name
    i = 0
    while os.path.exists('{}({:d}).png'.format(filename, i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(filename, i))


def plot3d(param1, param2, z, title='', path=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.meshgrid(param2.values, param1.values)
    ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contourf(x, y, z, zdir='z', offset=0, cmap=CM.coolwarm)
    cset = ax.contourf(x, y, z, zdir='x', offset=param2.range_min, cmap=CM.coolwarm)
    cset = ax.contourf(x, y, z, zdir='y', offset=param1.range_max, cmap=CM.coolwarm)
    ax.set_title(title)
    ax.set_xlabel(param2.name + ' ' + param2.unit)
    ax.set_ylabel(param1.name + ' ' + param1.unit)
    filename = path + title + '_' + param1.name + str(param1.resolution) + '_' + \
               param2.name + str(param2.resolution) + '_'
    i = 0
    while os.path.exists('{}({:d}).png'.format(filename, i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(filename, i))


def marginal_plot(param, path=''):
    """
    Plotting RandomVariable type

    :param param: RandomVariable type
    :param path: working path
    :return: Plots marginal posteriors and likelihoods
    """
    check_directory(path + "/marginal")

    # Check posterior distribution correctness
    print "The MARGINAL posterior is probability distributions? Integrate(posterior) = " + str(
        np.sum(param.posterior) * param.step)

    # Plot posterior
    plt.figure(figsize=(12,8))
    plt.grid()
    plt.title(param.name + " posterior (r) and prior (b) distribution")
    plt.xlabel(param.name + ' ' + param.unit)
    plt.ylabel("p")
    plt.plot(param.values, param.posterior,  'o', color="#FF5F2E", label="posterior")
    plt.plot(param.values, param.prior, color='#2FA5A0', label="prior")

    tt = np.linspace(param.range_min, param.range_max, 2000)
    fitted = normal(tt, param.fitted_gauss[0][0], param.fitted_gauss[0][1])
    plt.plot(tt, fitted, color="#FF5F2E", label="fitted posterior")
    plt.axvline(param.value, color='#34A52F', label="true value", linestyle='dashed')
    plt.axvline(param.max_p, color='#FF1493', label="max inferred", linestyle='dotted')
    plt.legend(loc='best', framealpha=0.4)
    filename = path + "/marginal/" + param.name + "_P"
    i = 0
    while os.path.exists('{}({:d}).png'.format(filename, i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(filename, i))
    print "Plot done! File path: " + filename

    # Plot likelihood
    plt.figure(figsize=(12, 8))
    plt.title(param.name + " likelihood distribution")
    plt.grid()
    plt.xlabel(param.name + ' ' + param.unit)
    plt.ylabel("p")
    plt.axvline(param.value, color='#34A52F', label="ture value", linestyle='dashed')
    plt.axvline(param.max_l, color='#FF1493', label="max inferred", linestyle='dotted')
    plt.plot(param.values, param.likelihood, marker='o', color="#ffc82e", label="likelihood")
    plt.legend(loc='best', framealpha=0.4)
    filename = path + "/marginal/" + param.name + "_L"
    i = 0
    while os.path.exists('{}({:d}).png'.format(filename, i)):
        i += 1
    plt.savefig('{}({:d}).png'.format(filename, i))
    print "Plot done! File path: " + filename


def plot_stat(stat, param, path='', bin=None):

    avrg_sigma = np.average(stat[:, 0])
    std_sigma = np.std(stat[:, 0])

    avrg_sigma_err = np.average(stat[:,1])

    avrg_rdiff = np.average(abs(stat[:, 2]))
    std_rdiff = np.std(abs(stat[:, 2]))

    avrg_acc = np.average(stat[:, 3])
    std_acc = np.std(stat[:, 3])

    avrg_sharp = np.average(stat[:, 4])
    std_sharp = np.std(stat[:, 4])

    avrg_broad = np.average(stat[:, 5])
    std_broad = np.std(stat[:, 5])

    # Plot illustration
    x = np.linspace(param.range_min, param.range_max,  2000, dtype=float)
    prior = normal(x, param.mean, param.sigma)

    posterior = normal(x, param.mean, avrg_sigma)
    post_max = np.amax(posterior)


    plt.figure(figsize=(12, 8))
    plt.title('rdiff(g): %.0f%%, acc(b): %.2f%%, gain: (%.2f pm %.2f), broad: (%.1f%% pm %.1f%%) |' % (
        avrg_rdiff, avrg_acc, avrg_sharp, std_sharp, avrg_broad, std_broad) + param.name)
    plt.xlabel(param.name + ' ' + param.unit)
    plt.ylabel('Probability')
    plt.grid(True)
    plt.plot(x, posterior, color='#9c3853', label="posterior")
    plt.plot(x, prior, color='#2FA5A0', label="prior")
    diff = avrg_rdiff/100*param.value
    std = std_rdiff/100*param.value
    plt.axvspan(param.mean - diff - std, param.mean + diff + std, facecolor='g', alpha=0.1)
    # plt.axvline(x=param.mean+avrg_diff, color='#389c81')
    # plt.axvline(x=param.mean-avrg_diff, color='#389c81')
    # plt.axhline(y=(avrg_acc/100)*post_max, xmin=0, xmax=1000, color='#38539C', linewidth=1)
    plt.axhspan((avrg_acc / 100) * post_max - (std_acc / 100) * post_max,
                (avrg_acc / 100) * post_max + (std_acc / 100) * post_max,
                facecolor='b', alpha=0.1)
    plt.legend()
    plt.savefig(path + "/illustration_" + param.name + ".png")

    # # Plot shape of posterior
    # max_p = normal(x, param.mean, avrg_sigma + std_sigma)
    # min_p = normal(x, param.mean, avrg_sigma - std_sigma)
    #
    # # Sharper than prior minimum value is 1
    # gain_bot = avrg_sharp-std_sharp
    # if gain_bot < 1:
    #     gain_bot = 1.
    #
    # plt.figure(figsize=(12,8))
    # plt.title("Sharpness of posterior")
    # plt.xlabel(param.name + ' ' + param.unit)
    # plt.ylabel('Probability')
    # plt.grid(True)
    # plt.plot(x, posterior, color='#9c3853', label="average: %.2f" % avrg_sharp)
    # plt.plot(x, max_p, color='#fff34d', label="avrg+std: %.2f" % (avrg_sharp+std_sharp))
    # plt.plot(x, min_p, color='#f9484f', label="avrg-std: %.2f" % (gain_bot))
    # plt.plot(x, prior, color='#2FA5A0', label="prior")
    #
    # plt.legend()
    # plt.savefig(path + "/plook_"+param.name+".png")


    # Plot histograms
    check_directory(path + "/histograms")
    if bin is None:
        bin = int(len(stat[:, 0]) / 2)

    plt.figure(figsize=(12, 8))
    plt.title("Relative deviation of true parameter | " + param.name + " " + str(param.mean))
    plt.xlabel(param.name + ' (average: %.1f%%)' % avrg_rdiff)
    plt.ylabel('Occurrence ')
    plt.grid(True)
    plt.hist(stat[:,2], bin, facecolor='#D44A4B', normed=False)
    plt.savefig(path + "/histograms/rdeviation_" + param.name + ".png")

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy | " + param.name)
    plt.xlabel("p_true/p_max" + ' (average: %.1f%% )' % avrg_acc)
    plt.ylabel('Occurrence')
    plt.grid(True)
    plt.hist(stat[:, 3], bin, facecolor='#3BA9A8', normed=False)
    plt.savefig(path + "/histograms/accuracy_" + param.name + ".png")

    plt.figure(figsize=(12, 8))
    plt.title("Posterior how many times sharper than prior | " + param.name)
    plt.xlabel("(prior sharpness)/(posterior sharpness)" + ' ' + '(avrg: %.2f )' % avrg_sharp)
    plt.ylabel('Occurrence')
    plt.grid(True)
    plt.hist(stat[:, 4], bin, facecolor='#4A4BD4', normed=False)
    plt.savefig(path + "/histograms/pSharpness_" + param.name + ".png")

    plt.figure(figsize=(12, 8))
    plt.title("Posterior broadness relative to prior | " + param.name)
    plt.xlabel("(posterior sharpness)/(prior sharpness)*100" + ' ' + '(avrg: %.1f%% )' % avrg_broad)
    plt.ylabel('Occurrence')
    plt.grid(True)
    plt.hist(stat[:, 5], bin, facecolor='#d4d34a', normed=False)
    plt.savefig(path + "/histograms/pBroadness_" + param.name + ".png")

    plt.figure(figsize=(12, 8))
    plt.title("Relative error of fitting | " + param.name)
    plt.xlabel("(sigma_err/sigma + mean_err/mean)*100" + ' ' + '(avrg: %.1f%% )' % avrg_sigma_err)
    plt.ylabel('Occurrence')
    plt.grid(True)
    plt.hist(stat[:, 1], bin, facecolor='#3ce1bb', normed=False)
    plt.savefig(path + "/histograms/fiterr_" + param.name + ".png")

    print "Stat plotted to: " + path

    # plt.figure()
    # plt.title("Fitted gaussian sigma parameter | " + param.name)
    # plt.xlabel('Sigma' + ' ' + '(average: %.2e )' % avrg_sigma)
    # plt.ylabel('Occurrence')
    # plt.grid(True)
    # plt.hist(stat[:, 1], bin, facecolor='#D44A4B', normed=False)
    # plt.savefig(path + "/sigma_"+param.name+".png")

if __name__ == '__main__':
    from module.probability import RandomVariable

    pRa = RandomVariable(name='Ra', range_min=50, range_max=150, resolution=40, mean=100., sigma=20)
    pgpas = RandomVariable(name='gpas', range_min=0.00005, range_max=0.00015, resolution=40, mean=0.0001, sigma=0.00002)
    pcm = RandomVariable(name='cm', range_min=0.5, range_max=1.5, resolution=40, mean=1., sigma=0.2)

    cm_stat = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/cm_stat.txt")
    Ra_stat = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/Ra_stat.txt")
    gpas_stat = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp/gpas_stat.txt")

    plot_stat(cm_stat, pcm, path="/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp")
    plot_stat(Ra_stat, pRa, path="/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp")
    plot_stat(gpas_stat, pgpas, path="/Users/Dani/TDK/parameter_estim/stim_protocol2/ramp")

