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
training, testing, m, n = import_raw(filename)	# import data 
end = time.time()
print('prepare time: ', end - start)

print('start training')
start = time.time()
np.random.seed(seed = None)

# first hidden layer h0 - autoencoder elm-ae
in_w0 = np.random.normal(size = (training['signal'].shape[1], 1200))	# randomize input weights of h0
in_w0, r = np.linalg.qr(in_w0)	# make orthogonal weight vectors
h0 = hidden(training['signal'], in_w0)		# output of h0
out_w0 = np.matmul(np.linalg.pinv(h0), training['signal'])	# tune output weight of h0
training['signal'] = np.matmul(h0, out_w0)	# the input signal after autoencoder

# second hidden layer h1 - elm
in_w1 = np.random.normal(size = (training['signal'].shape[1], 4100))	# randomize input weights of h1
h1 = hidden(training['signal'], in_w1)		# output of h1
# out_w1 = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
out_w1 = np.matmul(np.linalg.pinv(h1), training['thickness'])	# tune output weight for h1
end = time.time()

print('training time: ', end - start)

print('start testing')
start = time.time()

# h0
h0 = hidden(testing['signal'], in_w0)	# output of h0
testing['signal'] = np.matmul(h0, out_w0)		# after autoencoder

# h1
h1 = hidden(testing['signal'], in_w1)
approx = np.matmul(h1, out_w1)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
sio.savemat('network.mat', {'in_w0': in_w0, 'out_w0': out_w0, 'in_w1': in_w1, 'out_w1': out_w1, 'mean': m, 'n': n})

# plot the first 100 samples 
print('plot')
plot_model(approx, testing['thickness'])

###########################################################################################
# # test mse
# n_nodes = 0	# number of node in the hidden layer
# error = np.array([[]])	# initialize matrix to store all the mse 

# while n_nodes < 4100:
# 	# start training the model
# 	n_nodes += 100
# 	start = time.time()
# 	in_w0 = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of the network 
# 	in_w0, r = np.linalg.qr(in_w0)
# 	# h = hidden(training['signal'], input_w)		# output of the hidden layer
# 	h0 = hidden(training['signal'], in_w0)		# output of the hidden layer
# 	out_w0 = np.matmul(np.linalg.pinv(h0), training['signal'])

# 	training['signal'] = np.matmul(h0, out_w0)
# 	in_w1 = np.random.normal(size = (training['signal'].shape[1], 4100))	# randomize input weights of the network 
# 	h1 = hidden(training['signal'], in_w1)		# output of the hidden layer
# 	# out_w1 = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
# 	out_w1 = np.matmul(np.linalg.pinv(h1), training['thickness'])
# 	end = time.time()
# 	print('training time ', n_nodes/100, ': ', end - start)

# 	# start testing
# 	start = time.time()
# 	h0 = hidden(testing['signal'], in_w0)	# output of the hidden layer
# 	testing['signal'] = np.matmul(h0, out_w0)		# approximated values
# 	h1 = hidden(testing['signal'], in_w1)
# 	approx = np.matmul(h1, out_w1)
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