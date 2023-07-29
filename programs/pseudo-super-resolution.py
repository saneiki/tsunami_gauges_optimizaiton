import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import subscript

obj = subscript.PSR(nmod=21)

list_opt_method = ['QR', 'QR', 'QR', 'RANDOM']
list_noise_type=['unit', 'unit', 'unit', 'unit'] #'unit' or 'random'
list_nopt = [7, 21, 42, 42]
for opt_method, noise_type, nopt in zip(list_opt_method, list_noise_type, list_nopt):
  obj.Waveform(om=opt_method, nt=noise_type, no=nopt)


list_opt_method = ['QR', 'GA', 'NOWPHAS', 'RANDOM', 'QR', 'QR', 'GA']
list_noise_type = ['unit', 'unit', 'unit', 'unit', 'random', 'unit', 'random']
list_test_noise = ['off', 'off', 'off', 'off', 'on', 'on', 'on']

for opt_method,noise_type, test_noise in zip(list_opt_method, list_noise_type, list_test_noise):
  obj.SRErrorVsGauge(om=opt_method, nt=noise_type, tn=test_noise)