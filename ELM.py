import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from model_func import *

# filename = 'data_10.mat'
# filename = 'data_5000.mat'
filename = 'data_10000_10.mat'

print('read data')
start = time.time()
training, testing = import_raw(filename)
end = time.time()
print('prepare time: ', end - start)

# start training the model
print('start training')
start = time.time()
np.random.seed(seed = None)
input_w = np.random.normal(size = (training['signal'].shape[1], 10000))	# randomize input weights of the network 
h = hidden(training['signal'], input_w)		# output of the hidden layer
output_w = np.linalg.lstsq(h, training['thickness'])[0]		# use least square to find the optimal output weight 
end = time.time()
print('training time: ', end - start)

# start testing
print('start testing')
h = hidden(testing['signal'], input_w)	# output of the hidden layer
approx = np.matmul(h, output_w)		# approximated values
error = mse(approx, testing['thickness'])	# mean square error
print('mse: ', error)

# plot the first 100 samples 
print('plot')
expected = plt.scatter(np.linspace(0, 99, 100), testing['thickness'][0:100], color = 'red')
approximated, = plt.plot(np.linspace(0, 99, 100), approx[0:100], color = 'black')
plt.legend([expected, approximated], ['Expected output', 'Approximated model'])
plt.xlabel('Signal index')
plt.ylabel('Thickness')
plt.show()

sio.savemat('network.mat', {'input_w': input_w, 'output_w': output_w})