import sys
# add the path to the directory containing the module
sys.path.append("../../")
import numpy as np
import tables as tb
from module.save_load import load_parameter_set
from module.analyze import Analyse
from matplotlib import pyplot as plt
from matplotlib import cm as CM

p_names = ['Ra', 'gpas', 'ffact']

# directories to be plotted together (arbitrary number)
dirs = ["/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_inference_221018/exp_160601cw/",
        "/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_inference_221014/exp_160601cw/"]

rep_idx = 10  # repetition index to plot

pinits = []
for d in dirs:
        pinits.append(tb.open_file(d + "paramsetup.hdf5", mode="r"))

# constructing parameter lists
p_lists = []
for p in pinits:
        curr_plist = []
        for idx in range(p.root.params_init.shape[0]):
                curr_plist.append(p.root.params_init[idx, :])
        p_lists.append(curr_plist)

p_sets = []
for p_list in p_lists:
        p_sets.append(load_parameter_set(p_list))

# load likelihoods
lldbs = []
for d in dirs:
        lldbs.append(tb.open_file(d + "/ll0.hdf5", mode="r"))

# compute results with analysis class
results = []
for i in range(len(lldbs)):
        results.append(Analyse(lldbs[i].root.ll[:, rep_idx], p_sets[i], "./single_plots" ))
        # print(results[i])

pnum = len(results[0].p.params)

f, ax = plt.subplots(pnum, pnum, figsize=(14, 9))
f.subplots_adjust(hspace=.1, wspace=.1)

for row in range(pnum):
        for col in range(pnum):
                # Marginal plots
                if row == col:
                        i = row
                        ax[i, i].grid()
                        ax[row, col].set_xlabel(results[0].p.params[i].name + ' ' + results[0].p.params[i].unit)
                        for idx, result in enumerate(results):
                            ax[row, col].plot(result.p.params[i].values, result.p.params[i].likelihood, marker='o', label="run-{}".format(idx))
                        ax[row, col].legend(loc='best')

                        if row != pnum - 1:
                                ax[row, col].ticklabel_format(axis="y", style="sci", scilimits=(-4, 4))
                                ax[row, col].tick_params(axis='y', which='both', left='off', right='on',
                                                         labelleft='off',
                                                         labelright='on')
                                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='off',
                                                         labelbottom='off',
                                                         labeltop='off')
                                ax[row, col].xaxis.set_label_position('top')

                        else:
                                ax[row, col].ticklabel_format(axis="both", style="sci", scilimits=(-4, 4))
                                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='on',
                                                         labelbottom='on',
                                                         labeltop='off')
                                ax[row, col].tick_params(axis='y', which='both', left='off', right='on',
                                                         labelleft='off',
                                                         labelright='on')
                                ax[row, col].set_xlabel(results[0].p.params[col].name + ' ' + results[0].p.params[col].unit)

                # Joint plots
                elif col < row:
                        likelihoods = []
                        for result in results:
                            likelihoods.append(np.copy(result.likelihood))
                        ax[row, col].grid()

                        # Marginalize if needed
                        if len(results[0].p.params) > 2:
                                idxs_to_marginalize = []
                                step = 1
                                for idx, item in enumerate(results[0].p.params):
                                        if idx != row and idx != col:
                                                idxs_to_marginalize.append(idx)
                                                step *= item.step

                                # print "Indecies to marginalize: ", idxs_to_marginalize
                                for i in range(len(likelihoods)):
                                    likelihoods[i] = np.sum(likelihoods[i], axis=tuple(idxs_to_marginalize)) * item.step

                        # Contour plot
                        for result, likelihood in zip(results, likelihoods):
                            y, x = np.meshgrid(result.p.params[row].values, result.p.params[col].values)
                            cs = ax[row, col].contour(x, y, likelihood, cmap=CM.rainbow)


                        if col == 0:
                                ax[row, col].set_ylabel(results[0].p.params[row].name + ' ' + results[0].p.params[row].unit)
                        # ax[row, col].clabel(cs, inline=1, fontsize=5)

                        # Set up labels
                        ax[row, col].tick_params(axis='y', which='both', left='off', right='off', labelleft='off',
                                                 labelright='off')
                        ax[row, col].tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off',
                                                 labeltop='off')

                        # bottombar
                        if row == pnum - 1:
                                ax[row, col].tick_params(axis='x', which='both', top='off', bottom='on',
                                                         labelbottom='on',
                                                         labeltop='off')
                                ax[row, col].set_xlabel(results[0].p.params[col].name + ' ' + results[0].p.params[col].unit)
                                ax[row, col].ticklabel_format(axis="both", style="sci", scilimits=(-4, 4))

                        # sidebar
                        if col == 0:
                                ax[row, col].tick_params(axis='y', which='both', left='on', right='off', labelleft='on',
                                                         labelright='off')
                                ax[row, col].ticklabel_format(axis="both", style="sci", scilimits=(-4, 4))
                else:
                        ax[row, col].set_axis_off()

plt.show()

# close hdf5 files
for p in pinits:
        p.close()

for l in lldbs:
        l.close()

print()