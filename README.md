# parameter inference

## Generated experimental trace

### with white noise
1. One compartment passive model parameter (membrane capacitance 'cm') inference: <br>
**code:**<br>
_white_noise_inf1.py_<br>
**result:**<br>
_wn1_

2. One compartment passive model parameter (membrane capacitance'cm' and passive conductance 'gpas' ) inference:<br>
**code:**<br>
_white_noise_inf2.py_<br>
**result:**<br>
_wn2_
 
3. Stick and Ball model parameter (axial resistance 'Ra' and passive conductance 'gpas') inference: <br>
**code:**<br>
_white_noise_inf3.py_<br>
**result:**<br>
_wn3_

### with colored noise
4. One compartment passive model parameter (membrane capacitance'cm' and passive conductance 'gpas' ) inference: <br>
**code:**<br>
_colored_noise_inf1.py_<br>
**result:**<br>
_cn1_

5. Stick and Ball model parameter (axial resistance 'Ra' and passive conductance 'gpas') inference.<br>
**code:**<br>
_colored_noise_inf2.py_<br>
**result:**<br>
_cn2_

## True experimental trace
6. Inference on true experimental data<br>
**code:**<br>
_expference.py_<br>
**result:**<br>