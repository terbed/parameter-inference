from module.protocol_test import plot_single_results, plot_combined_results, mult_likelihood,\
    combine_likelihood, protocol_comparison
import time
import tables as tb

startTime = time.time()
nfp = 10  # Number of fixed parameters
p_names = ['gpas', 'k']

# Load parameter space initializator
pinit = tb.open_file(".../paramsetup.hdf5", mode="r")

# Plot some single results
plot_single_results(path=".../steps/3", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path=".../steps/20", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path=".../steps/200", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path=".../sins/1", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path=".../sins/10", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path=".../sins/100", numfp=nfp, which=0, dbs=pinit)


# Multiply likelihoods for each fixed parameter
mult_likelihood(path=".../sins/100", numfp=10, num_mult=30)
mult_likelihood(path=".../sins/10", numfp=10, num_mult=30)
mult_likelihood(path=".../sins/1", numfp=10, num_mult=30)
mult_likelihood(path=".../steps/3", numfp=10, num_mult=30)
mult_likelihood(path=".../steps/20", numfp=10, num_mult=30)
mult_likelihood(path=".../steps/200", numfp=10, num_mult=30)

# Create combine path_lists:
steps_list = [".../steps/3",
             ".../steps/20",
             ".../steps/200"]

sins_list = [".../sins/1",
             ".../sins/10",
             ".../sins/100"]

# Create combinations
combine_likelihood(sins_list, numfp=10, num_mult_single=10,
                   out_path=".../sins/comb")
combine_likelihood(steps_list, numfp=10, num_mult_single=10,
                   out_path=".../steps/comb")

plot_combined_results(".../sins/100", 10, dbs=pinit)
plot_combined_results(".../sins/10", 10, dbs=pinit)
plot_combined_results(".../sins/1", 10, dbs=pinit)
plot_combined_results(".../sins/comb", 10, dbs=pinit)
plot_combined_results(".../steps/3", 10, dbs=pinit)
plot_combined_results(".../steps/20", 10, dbs=pinit)
plot_combined_results(".../steps/200", 10, dbs=pinit)
plot_combined_results(".../steps/comb", 10, dbs=pinit)

path_list = [".../steps/3",
             ".../steps/20",
             ".../steps/200",
             ".../sins/1",
             ".../sins/10",
             ".../sins/100",
             ".../steps/comb",
             ".../sins/comb"]


protocol_comparison(path_list, numfp=10, inferred_params=p_names,
                    out_path="...", dbs=pinit)

pinit.close()
runningTime = (time.time()-startTime)/60
print "\n\nThe script was running for %f minutes" % runningTime
