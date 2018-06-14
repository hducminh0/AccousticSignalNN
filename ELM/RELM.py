import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from model_func import *

# filename = 'data_10_100.mat'
# filename = 'data_5000.mat'
# filename = 'data_10000_50_vf.mat'
filename = 'data_10000_50_vf_8layers.mat'

print('read data')
start = time.time()
training, testing, m, n = import_raw(filename)	# import data 
end = time.time()
print('prepare time: ', end - start)

print('start training')
start = time.time()
np.random.seed(seed = None)
# first hidden layer h0 random mapping
rm = np.random.normal(size = (training['signal'].shape[1], 10000))	# randomize input weights of h0
rm, r = np.linalg.qr(rm)	# make orthogonal weight vectors
training['signal'] = hidden(training['signal'], rm)		# output of h0
# second hidden layer h1 - elm
w1_in = np.random.normal(size = (training['signal'].shape[1], 500))	# randomize input weights of h1
h1 = hidden(training['signal'], w1_in)		# output of h1
# w1_out = np.linalg.lstsq(h1, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
w1_out = np.matmul(np.linalg.pinv(h1), training['thickness'])	# tune output weight for h1
end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
# h0
testing['signal'] = hidden(testing['signal'], rm)	# output of h0
# h1
h1 = hidden(testing['signal'], w1_in)
approx = np.matmul(h1, w1_out)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
sio.savemat('network.mat', {'rm': rm, 'w1_in': w1_in, 'w1_out': w1_out, 'mean': m, 'n': n})

# plot the first 100 samples 
print('plot')
approx = approx * n + m
testing['thickness'] = testing['thickness'] * n + m
plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])

###########################################################################################
# test mse
# n_nodes = 0	# number of node in the hidden layer
# n = 100
# error = np.array([[]])	# initialize matrix to store all the mse 

# while n_nodes < 4100:
# 	# start training the model
# 	n_nodes += n
# 	start = time.time()
# 	np.random.seed(seed = None)
# 	# first hidden layer h0 random mapping
# 	rm = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of h0
# 	rm, r = np.linalg.qr(rm)	# make orthogonal weight vectors
# 	training['signal'] = hidden(training['signal'], rm)		# output of h0
# 	# second hidden layer h1 - elm
# 	w1_in = np.random.normal(size = (training['signal'].shape[1], 4100))	# randomize input weights of h1
# 	h1 = hidden(training['signal'], w1_in)		# output of h1
# 	# w1_out = np.linalg.lstsq(h1, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
# 	w1_out = np.matmul(np.linalg.pinv(h1), training['thickness'])	# tune output weight for h1
# 	end = time.time()
# 	print('training time ', n_nodes/n, ': ', end - start)

# 	# start testing
# 	start = time.time()
# 	# h0
# 	testing['signal'] = hidden(testing['signal'], rm)	# output of h0
# 	# h1
# 	h1 = hidden(testing['signal'], w1_in)
# 	approx = np.matmul(h1, w1_out)
# 	error = np.insert(error, error.shape[1], mse(approx, testing['thickness']), axis = 1)	# mean square error
# 	end = time.time()
# 	print('testing time: ', end - start)

# # plot the mse 
# sio.savemat('error.mat', {'error': error})
# print('min mse index: ', np.argmin(error))
# plt.plot(np.linspace(n, n_nodes, n_nodes/n), error[0, :])
# plt.xlabel('Number of nodes')
# plt.ylabel('Mean square error')
# plt.show()
#################################################################################################
# error = np.array([[]])
# for i in range(0, 10):
# 	print('start training')
# 	start = time.time()
# 	np.random.seed(seed = None)
# 	# first hidden layer h0 random mapping
# 	rm = np.random.normal(size = (training['signal'].shape[1], 1200))	# randomize input weights of h0
# 	rm, r = np.linalg.qr(rm)	# make orthogonal weight vectors
# 	h0 = hidden(training['signal'], rm)		# output of h0
# 	# second hidden layer h1 - elm
# 	w1_in = np.random.normal(size = (h0.shape[1], 4100))	# randomize input weights of h1
# 	h1 = hidden(h0, w1_in)		# output of h1
# 	# w1_out = np.linalg.lstsq(h1, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
# 	w1_out = np.matmul(np.linalg.pinv(h1), training['thickness'])	# tune output weight for h1
# 	end = time.time()
# 	print('training time: ', end - start)

# 	print('start testing')
# 	start = time.time()
# 	# h0
# 	h0 = hidden(testing['signal'], rm)	# output of h0
# 	# h1
# 	h1 = hidden(h0, w1_in)
# 	approx = np.matmul(h1, w1_out)
# 	error = np.insert(error, error.shape[1], mse(approx, testing['thickness']), axis = 1)
# 	print('mse: ', error)
# 	end = time.time()
# 	print('testing time: ', end - start)

# print(np.std(error))
# print(np.mean(error))