import numpy as np
from module.load import load_inference
from plot import fullplot

duration = [3, 20, 200, 300]   # The last one is combination
p_num = 10

result_Ra = np.empty((len(duration), p_num+1))
result_cm = np.empty((len(duration), p_num+1))
result_gpas = np.empty((len(duration), p_num+1))

result_Ra_b = np.empty((len(duration), p_num+1))
result_cm_b = np.empty((len(duration), p_num+1))
result_gpas_b = np.empty((len(duration), p_num+1))

result_Ra_d = np.empty((len(duration), p_num+1))
result_cm_d = np.empty((len(duration), p_num+1))
result_gpas_d = np.empty((len(duration), p_num+1))

for idx, dur in enumerate(duration):
    result_Ra[idx, 0] = dur # Set up first col
    result_cm[idx, 0] = dur
    result_gpas[idx, 0] = dur
    result_Ra_b[idx, 0] = dur # Set up first col
    result_cm_b[idx, 0] = dur
    result_gpas_b[idx, 0] = dur
    result_Ra_d[idx, 0] = dur
    result_cm_d[idx, 0] = dur
    result_gpas_d[idx, 0] = dur
    for n in range(p_num):
        l = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)/loglikelihood.txt"
                       % (dur, n))
        Ra = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/3(%i)/fixed_params/Ra(0).txt"
                        %(n), dtype=str)
        cm = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/3(%i)/fixed_params/cm(0).txt"
                        %(n), dtype=str)
        gpas = np.loadtxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/3(%i)/fixed_params/gpas(0).txt"
                          %(n), dtype=str)

        inf = load_inference(l,"/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/%i(%i)" %(dur,n), Ra, cm, gpas)
        inf.run_evaluation()
        stat = inf.analyse_result()
        fullplot(inf)

        print stat[0][4]
        result_Ra[idx, n+1] = stat[0][4]
        result_Ra_b[idx, n + 1] = stat[0][5]
        result_cm[idx, n+1] = stat[1][4]
        result_cm_b[idx, n + 1] = stat[1][5]
        result_gpas[idx, n+1] = stat[2][4]
        result_gpas_b[idx, n + 1] = stat[2][5]
        result_Ra_d[idx, n+1] = stat[0][2]
        result_cm_d[idx, n + 1] = stat[1][2]
        result_gpas_d[idx, n + 1] = stat[2][2]

np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/Ra_sharpness.txt", result_Ra)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/Ra_broadness.txt", result_Ra_b)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/cm_sharpness.txt", result_cm)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/cm_broadness.txt", result_cm_b)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/gpas_sharpness.txt", result_gpas)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/gpas_broadness.txt", result_gpas_b)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/Ra_rdiff.txt", result_Ra_d)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/cm_rdiff.txt", result_cm_d)
np.savetxt("/Users/Dani/TDK/parameter_estim/stim_protocol2/combinig/steps/gpas_rdiff.txt", result_gpas_d)

