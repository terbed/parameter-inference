from module.evaluate import Evaluation


nfp = 10  # Number of fixed parameters
nr = 30   # Number of repetition
p_names = ['Ra', 'gpas']

dir = "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored_exp_Ra_gpas/"
subdirs = ["steps/3", "steps/20", "steps/200", "sins/1", "sins/10", "sins/100"]    # subdirectories of the man directory
comb_lists = [["steps/3", "steps/20", "steps/200"],         # we will combine these protocols
              ["sins/1", "sins/10", "sins/100"]]

e = Evaluation(nfp, nr, p_names, dir, subdirs, comb_lists)

# 1.) some single result plot
e.single_result_plot(which=0)

# 2.) likelihood multiplications
e.likelihood_mult()

# 3.) protocol combinations
e.combinations(comb_lists, ["steps/comb", "sins/comb"])

# 4.) protocol comparison
path_to_protocols = ["steps/3", "steps/20", "steps/200",         # we will combine these protocols
                     "sins/1", "sins/10", "sins/100",
                     "steps/comb", "sins/comb"]
xtick_list = ['3ms', '20ms', '200ms', '1Hz', '10Hz', '100Hz', 'steps comb', 'sins comb']
e.compare_protocols(pathlist=path_to_protocols, xticks=xtick_list)

