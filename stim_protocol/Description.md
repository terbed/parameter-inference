# Experiment information content
We try to __describe and compare the accuracy of electrophysiological experiments__. In this part we overview the effects of
different electrode stimulus toward the gain of information. To achieve this we apply two different stimulus, one with a
`narrow peak` of electrode current and other with a `broad length` and also with both of them.

#### Expectations:
- [x] With __narrow peak stimulus__ we will infer `cm` parameter as wall as with broad length stimulus, because this parameter is responsible
for transient behaviour of membrane potential.
- [X] With a __narrow peak stimulus__ we will infer the `gpas` parameter worse, because this parameter is responsible for the height
of trace after impregnation (with narrow stimulus there may not be "impregnation").
- [x] With __two stimulus__ we will improve the inference of `cm` parameter due to more transient data point. 
_(gpas also improved a little bit)_
- [X] Trivial expectation: increasing __data sampling frequency__ will increase the effectiveness of inference.
_(timestep (dt) increased from 0.1 to 0.05 caused twice accuracy in inference)_

#### Stimulus types:
1. broad length
2. narrow length
3. both

The neuron voltage response for the given stimulus (deterministic data from model) can be seen here in order:

## 1.1 One compartment model and white noise
_Directory: ow_


## 1.2 One compartment model and colored noise
_Directory: oc_

## 2.1 Ball and Stick model with white noise
_Directory: mw_

## 2.2 Ball and Stick model with colored noise
_Directory: mc_

## 3 Experimental data with white noise
_Directory: expw_