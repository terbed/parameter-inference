# parameter inference

## Generated experimental trace (synthetic simulation)

### with white noise
1. One compartment passive model parameter (membrane capacitance 'cm') inference: <br>
**code:**<br>
_white_noise_inf1.py_<br>
**result:**<br>
_wn1/statistics_

2. One compartment passive model parameter (membrane capacitance'cm' and passive conductance 'gpas' ) inference:<br>
**code:**<br>
_white_noise_inf2.py_<br>
**result:**<br>
_wn2/statistics_
 
3. Stick and Ball model parameter (axial resistance 'Ra' and passive conductance 'gpas') inference: <br>
**code:**<br>
_white_noise_inf3.py_<br>
**result:**<br>
_wn3/statistics_

4. The effect of sampling parameters from uniform or prior distribution and its resolution on inference: <br>
_Parameter sampling test_

### with colored noise
1. One compartment passive model parameter (membrane capacitance'cm' and passive conductance 'gpas' ) inference: <br>
**code:**<br>
_colored_noise_inf1.py_<br>
**result:**<br>
_cn1/statistics_

2. Stick and Ball model parameter (axial resistance 'Ra' and passive conductance 'gpas') inference.<br>
**code:**<br>
_colored_noise_inf2.py_<br>
**result:**<br>
_cn2/statistics_

3. The simulation timestep effect on the inference:<br>
_Simulation time resolution test_

## True experimental trace
Inference on true experimental data<br>
**code:**<br>
_expference.py_<br>
**result:**<br>
_exp_