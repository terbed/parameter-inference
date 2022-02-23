from module.protocol_test import plot_single_results, plot_combined_results, mult_likelihood,\
    combine_likelihood, protocol_comparison
import time
import tables as tb

startTime = time.time()
nfp = 10  # Number of fixed parameters
p_names = ['gpas', 'k']

# Load parameter space initializator
pinit = tb.open_file("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/paramsetup.hdf5", mode="r")

# Plot some single results
plot_single_results(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/3", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/20", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/200", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/1", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/10", numfp=nfp, which=0, dbs=pinit)
plot_single_results(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/100", numfp=nfp, which=0, dbs=pinit)


# Multiply likelihoods for each fixed parameter
mult_likelihood(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/100", numfp=10, num_mult=30)
mult_likelihood(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/10", numfp=10, num_mult=30)
mult_likelihood(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/1", numfp=10, num_mult=30)
mult_likelihood(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/3", numfp=10, num_mult=30)
mult_likelihood(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/20", numfp=10, num_mult=30)
mult_likelihood(path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/200", numfp=10, num_mult=30)

# Create combine path_lists:
steps_list = ["/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/3",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/20",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/200"]

sins_list = ["/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/1",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/10",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/100"]

# Create combinations
combine_likelihood(sins_list, numfp=10, num_mult_single=10,
                   out_path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/comb")
combine_likelihood(steps_list, numfp=10, num_mult_single=10,
                   out_path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/comb")

plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/100", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/10", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/1", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/comb", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/3", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/20", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/200", 10, dbs=pinit)
plot_combined_results("/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/comb", 10, dbs=pinit)

path_list = ["/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/3",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/20",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/200",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/1",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/10",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/100",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/steps/comb",
             "/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens/sins/comb"]


protocol_comparison(path_list, numfp=10, inferred_params=p_names,
                    out_path="/home/szabolcs/parameter_inference/stim_protocol2_v24/comb_colored_soma_gpas-dens", dbs=pinit)

pinit.close()
runningTime = (time.time()-startTime)/60
print("\n\nThe script was running for %f minutes" % runningTime)
