import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from model_func import *

# filename = 'data_10.mat'
# filename = 'data_5000.mat'
# filename = 'data_10000_50_vf.mat'
# filename = 'data_10000_50_vf_8layers.mat'
filename = 'data_10000_100_8layers.mat'

print('read data')
start = time.time()
training, testing, m, n = import_raw(filename)	# import data 
end = time.time()
print('prepare time: ', end - start)

print('start training')
start = time.time()
np.random.seed(seed = None)
# # random mapping
# rm, r = np.linalg.qr(np.random.normal(size = (training['signal'].shape[1], 100)))	# randomize input weights of h0
# training['signal'] = hidden(training['signal'], rm)
# first hidden layer h0 - autoencoder elm-ae
w0_in = np.random.normal(size = (training['signal'].shape[1], 8000))	# randomize input weights of h0
w0_in, r = np.linalg.qr(w0_in)	# make orthogonal weight vectors
h0 = hidden(training['signal'], w0_in)		# output of h0
w0_out = np.matmul(np.linalg.pinv(h0), training['signal'])	# tune output weight of h0
training['signal'] = np.matmul(h0, w0_out)	# the input signal after autoencoder
# second hidden layer h1 - elm
w1_in = np.random.normal(size = (training['signal'].shape[1], 4700))	# randomize input weights of h1
h1 = hidden(training['signal'], w1_in)		# output of h1
# w1_out = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
w1_out = np.matmul(np.linalg.pinv(h1), training['thickness'])	# tune output weight for h1
end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
# # random mapping
# testing['signal'] = hidden(testing['signal'], rm)
# h0
h0 = hidden(testing['signal'], w0_in)	# output of h0
testing['signal'] = np.matmul(h0, w0_out)		# after autoencoder
# h1
h1 = hidden(testing['signal'], w1_in)
approx = np.matmul(h1, w1_out)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
sio.savemat('network.mat', {'w0_in': w0_in, 'w0_out': w0_out, 'w1_in': w1_in, 'w1_out': w1_out, 'mean': m, 'n': n})

# plot the first 100 samples 
print('plot')
approx = approx * n + m
testing['thickness'] = testing['thickness'] * n + m
plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])

###########################################################################################
# # test mse
# n_nodes = 0	# number of node in the hidden layer
# error = np.array([[]])	# initialize matrix to store all the mse 

# while n_nodes < 7000:
# 	# start training the model
# 	n_nodes += 100
# 	start = time.time()
# 	# random mapping
# 	rm, r = np.linalg.qr(np.random.normal(size = (training['signal'].shape[1], 100)))	# randomize input weights of h0
# 	training['signal'] = hidden(training['signal'], rm)
# 	w0_in = np.random.normal(size = (training['signal'].shape[1], 100))	# randomize input weights of the network 
# 	w0_in, r = np.linalg.qr(w0_in)
# 	# h = hidden(training['signal'], input_w)		# output of the hidden layer
# 	h0 = hidden(training['signal'], w0_in)		# output of the hidden layer
# 	w0_out = np.matmul(np.linalg.pinv(h0), training['signal'])

# 	training['signal'] = np.matmul(h0, w0_out)
# 	w1_in = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of the network 
# 	h1 = hidden(training['signal'], w1_in)		# output of the hidden layer
# 	# w1_out = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
# 	w1_out = np.matmul(np.linalg.pinv(h1), training['thickness'])
# 	end = time.time()
# 	print('training time ', n_nodes/100, ': ', end - start)

# 	# start testing
# 	start = time.time()# random mapping
# 	testing['signal'] = hidden(testing['signal'], rm)
# 	h0 = hidden(testing['signal'], w0_in)	# output of the hidden layer
# 	testing['signal'] = np.matmul(h0, w0_out)		# approximated values
# 	h1 = hidden(testing['signal'], w1_in)
# 	approx = np.matmul(h1, w1_out)
# 	error = np.insert(error, error.shape[1], mse(approx, testing['thickness']), axis = 1)	# mean square error
# 	end = time.time()
# 	print('testing time: ', end - start)

# # plot the mse 
# sio.savemat('error.mat', {'error': error})
# plt.plot(np.linspace(100, n_nodes, n_nodes/100), error[0, :])
# plt.xlabel('Number of nodes')
# plt.ylabel('Mean square error')
# plt.show()
#################################################################################################
