# Single runs
- (0) Noise: 7[mV], resolution: 80
- (1) Noise: 7[mV], resolution: 40
- (2) Noise: 7[mV], resolution: 40
- (3) multiplied 50 (2) likelihood solution
- (4) Noise: 1/sqrt(50)*7 [mV], resolution: 40
- (5) Repetition of (1) and (2)
. (6) Repetition of (1), (2) and (5)
- (7) Noise: 7[mV], resolution: 160
- (8) Noise: 7[mV], resolution: 160, with range: Ra[40,300], cm[0.5,1.7], gpas[0.00005, 0.00015]
- (9) (8) repetition (saved synthetic trace)
- (10) with (9) synthetic trace in range: Ra[40,150], ~

## Averaging data vs inference on each set
With (3) and (4) we test the hypothesis: 
1) Inference on averaged data
2) Inference on every single data set and then multiply likelihoods

The assertion is that 1. and 2. yields the same result.

