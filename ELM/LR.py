import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from model_func import *

filename = 'data_10000_100_vcyo_8layers.mat'
# filename = 'data_10000_75_vcyo_8layers.mat'
# filename = 'data_10000_50_vcyo_8layers.mat'
# filename = 'data_10000_25_vcyo_8layers.mat'

# filename = 'data_10000_100_vf_8layers.mat'
# filename = 'data_10000_75_vf_8layers.mat'
# filename = 'data_10000_50_vf_8layers.mat'
# filename = 'data_10000_25_vf_8layers.mat'

# filename = 'data_10000_100_8layers.mat'
# filename = 'data_10000_75_8layers.mat'
# filename = 'data_10000_50_8layers.mat'
# filename = 'data_10000_25_8layers.mat'

print('read data')
start = time.time()
training, testing, m, n = import_raw(filename)
end = time.time()
print('prepare time: ', end - start)

print('start training')
start = time.time()
w_out = np.linalg.lstsq(training['signal'], training['thickness'], rcond = None)[0]		# use pseudo inverse to find the optimal output weight 
end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
approx = np.matmul(testing['signal'], w_out)		# approximated values
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
# sio.savemat('network.mat', {'w_in': w_in, 'w_out': w_out, 'mean': m, 'n': n})

# plot the first 100 samples 
print('plot')
approx = approx * n + m
testing['thickness'] = testing['thickness'] * n + m
plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])