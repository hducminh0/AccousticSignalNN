import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from model_func import *

# filename = 'data_10.mat'
# filename = 'data_5000.mat'
filename = 'data_10000_50_vf.mat'

print('read data')
start = time.time()
training, testing, m, n = import_raw(filename)
end = time.time()
print('prepare time: ', end - start)

n_nodes = 4700	# number of node in the hidden layer
print('start training')
start = time.time()
np.random.seed(seed = None)
input_w = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of the network 
h = hidden(training['signal'], input_w)		# output of the hidden layer
output_w = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
# output_w = np.matmul(np.linalg.pinv(h), training['thickness'])
end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
h = hidden(testing['signal'], input_w)	# output of the hidden layer
approx = np.matmul(h, output_w)		# approximated values
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
sio.savemat('network.mat', {'input_w': input_w, 'output_w': output_w, 'mean': m, 'n': n})

# plot the first 100 samples 
print('plot')
plot_model(approx, testing['thickness'])

###########################################################################################
# # test mse
# n_nodes = 100	# number of node in the hidden layer
# error = np.array([[]])	# initialize matrix to store all the mse 

# while n_nodes <= training['signal'].shape[0]:
# 	# start training the model
# 	print('start training')
# 	start = time.time()
# 	np.random.seed(seed = None)
# 	input_w = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of the network 
# 	h = hidden(training['signal'], input_w)		# output of the hidden layer
# 	output_w = np.matmul(np.linalg.pinv(h), training['thickness'])
# 	end = time.time()
# 	print('training time ', n_nodes/100, ': ', end - start)

# 	# start testing
# 	print('start testing')
# 	start = time.time()
# 	h = hidden(testing['signal'], input_w)	# output of the hidden layer
# 	approx = np.matmul(h, output_w)		# approximated values
# 	error = np.insert(error, error.shape[1], mse(approx, testing['thickness']), axis = 1)	# mean square error
# 	n_nodes += 100
# 	end = time.time()
# 	print('testing time: ', end - start)

# # plot the mse 
# sio.savemat('error.mat', {'error': error})
# plt.plot(np.linspace(100, n_nodes - 100, (n_nodes - 100)/100 - 1), error[0, 0:69])
# plt.xlabel('Number of nodes')
# plt.ylabel('Mean square error')
# plt.show()
#################################################################################################