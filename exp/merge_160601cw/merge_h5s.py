import sys
# add the path to the directory containing the module
sys.path.append("../../")
import numpy as np
import tables as tb
from module.save_load import load_parameter_set
from module.analyze import Analyse
from matplotlib import pyplot as plt
from matplotlib import cm as CM

p2merge = 'gpas'

# directories to be plotted together (arbitrary number)
dirs = ["/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_inference_221018/exp_160601cw/",
        "/Users/admin/PROJECTS/SPE/parameter-inference/exp/exp_inference_221014/exp_160601cw/"]

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
rep_idx = 0
likelihoods = []
for i in range(len(lldbs)):
        likelihoods.append(np.reshape(lldbs[i].root.ll[:, rep_idx], p_sets[i].shape))


# get relevant parameter information
axis = 0
for p_list in p_lists:
        for p in p_list:
                if p_list[0] == p2merge:
                        


print("")
