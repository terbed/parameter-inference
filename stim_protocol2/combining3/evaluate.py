from module.protocol_test import plot_single_results, plot_combined_results, mult_likelihood,\
    combine_likelihood, protocol_comparison
import time
import tables as tb

startTime = time.time()
nfp = 10  # Number of fixed parameters

# Load parameter space initializator
pinit = tb.open_file("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/paramsetup.hdf5", mode="r")

# Plot some single results
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/20", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/200", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/1", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/10", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100", numfp=nfp, which=0, dbs=pinit)


# Multiply likelihoods for each fixed parameter
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/10", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/1", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/20", numfp=10, num_mult=30)
mult_likelihood(path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/200", numfp=10, num_mult=30)

# Create combine path_lists:
steps_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/20",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/200"]

zaps_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/1",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/10",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100"]

# Create combinations
combine_likelihood(zaps_list, numfp=10, num_mult_single=10,
                   out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/comb")
combine_likelihood(steps_list, numfp=10, num_mult_single=10,
                   out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/comb")

plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/10", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/1", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/comb", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/20", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/200", 10, dbs=pinit)
plot_combined_results("/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/comb", 10, dbs=pinit)

path_list = ["/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/3",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/20",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/200",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/1",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/10",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/100",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/steps/comb",
             "/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3/zaps/comb"]


protocol_comparison(path_list, numfp=10, inferred_params=['Ra', 'cm', 'gpas'],
                    out_path="/Users/Dani/TDK/parameter_estim/stim_protocol2/combining3", dbs=pinit)

pinit.close()
runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime