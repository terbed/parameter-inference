# parameter inference

## Generated experimental trace

### with white noise
1. One compartment passive model parameter (membrane capacitance 'cm') inference: 
**code:**
_white_noise_inf1.py_
**result:**
_wn1_

2. One compartment passive model parameter (membrane capacitance'cm' and passive conductance 'gpas' ) inference:
**code:**
_white_noise_inf2.py_
**result:**
_wn2_
 
3. Stick and Ball model parameter (axial resistance 'Ra' and passive conductance 'gpas') inference: 
**code:**
_white_noise_inf3.py_
**result:**
_wn3_

### with colored noise
4. One compartment passive model parameter (membrane capacitance'cm' and passive conductance 'gpas' ) inference:
**code:**
_colored_noise_inf1.py_
**result:**
_cn1_

5. Stick and Ball model parameter (axial resistance 'Ra' and passive conductance 'gpas') inference.
**code:**
_colored_noise_inf2.py_
**result:**
_cn2_

## True experimental trace
6. Inference on true experimental data
**code:**
_expference.py_
**result:**