from module.protocol_test import plot_single_results, plot_combined_results, mult_likelihood,\
    combine_likelihood, protocol_comparison
import time
import tables as tb

startTime = time.time()
nfp = 10  # Number of fixed parameters
p_names = ['Ra', 'gpas']

# Load parameter space initializator
pinit = tb.open_file("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/paramsetup.hdf5", mode="r")

# Plot some single results
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/3", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/20", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/200", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/1", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/10", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/100", numfp=nfp, which=0, dbs=pinit)


# Multiply likelihoods for each fixed parameter
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/100", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/10", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/1", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/3", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/20", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/200", numfp=10, num_mult=30)

# Create combine path_lists:
steps_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/3",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/20",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/200"]

sins_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/1",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/10",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/100"]

# Create combinations
combine_likelihood(sins_list, numfp=10, num_mult_single=10,
                   out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/comb")
combine_likelihood(steps_list, numfp=10, num_mult_single=10,
                   out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/comb")

plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/100", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/10", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/1", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/comb", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/3", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/20", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/200", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/comb", 10, dbs=pinit)

path_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/3",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/20",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/200",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/1",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/10",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/100",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/steps/comb",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored/sins/comb"]


protocol_comparison(path_list, numfp=10, inferred_params=p_names,
                    out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining_colored", dbs=pinit)

pinit.close()
runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime