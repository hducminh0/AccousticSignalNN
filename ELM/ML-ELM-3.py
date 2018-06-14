import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from model_func import *

# filename = 'data_10.mat'
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

# first hidden layer h0 - autoencoder elm-ae
w0_in = np.random.normal(size = (training['signal'].shape[1], 20))	# randomize input weights of h0
w0_in, r = np.linalg.qr(w0_in)	# make orthogonal weight vectors
h0 = hidden(training['signal'], w0_in)		# output of h0
w0_out = np.matmul(np.linalg.pinv(h0), training['signal'])	# tune output weight of h0
training['signal'] = np.matmul(h0, w0_out)	# the input signal after autoencoder
# second hidden layer h1 - elm-ae
w1_in = np.random.normal(size = (training['signal'].shape[1], 20))	# randomize input weights of h1
w1_in, r = np.linalg.qr(w1_in)	# make orthogonal weight vectors
h1 = hidden(training['signal'], w1_in)		# output of h1
w1_out = np.matmul(np.linalg.pinv(h1), training['signal'])	# tune output weight of h1
training['signal'] = np.matmul(h1, w1_out)	# the input signal after autoencoder
# third hidden layer h2 - elm
w2_in = np.random.normal(size = (training['signal'].shape[1], 10))	# randomize input weights of h2
h2 = hidden(training['signal'], w2_in)		# output of h2
# w2_out = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
w2_out = np.matmul(np.linalg.pinv(h2), training['thickness'])	# tune output weight for h2
end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
# h0
h0 = hidden(testing['signal'], w0_in)	# output of h0
testing['signal'] = np.matmul(h0, w0_out)		# after autoencoder
# h1
h1 = hidden(testing['signal'], w1_in)	# output of h0
testing['signal'] = np.matmul(h1, w1_out)		# after autoencoder
# h2
h2 = hidden(testing['signal'], w2_in)
approx = np.matmul(h2, w2_out)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
sio.savemat('network.mat', {'w0_in': w0_in, 'w0_out': w0_out, 'w1_in': w1_in, 'w1_out': w1_out, 'w2_in': w2_in, 'w2_out': w2_out, 'mean': m, 'n': n})

# plot the first 100 samples 
print('plot')
approx = approx * n + m
testing['thickness'] = testing['thickness'] * n + m
plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])


###########################################################################################
# # test mse
# n_nodes = 0	# number of node in the hidden layer
# error = np.array([[]])	# initialize matrix to store all the mse 

# startto = time.time()
# while n_nodes < 3100:
# 	# start training the model
# 	n = 100
# 	n_nodes += n
# 	start = time.time()
# 	np.random.seed(seed = None)
# 	# first hidden layer h0 - autoencoder elm-ae
# 	w0_in = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of h0
# 	w0_in, r = np.linalg.qr(w0_in)	# make orthogonal weight vectors
# 	h0 = hidden(training['signal'], w0_in)		# output of h0
# 	w0_out = np.matmul(np.linalg.pinv(h0), training['signal'])	# tune output weight of h0
# 	training['signal'] = np.matmul(h0, w0_out)	# the input signal after autoencoder

# 	# second hidden layer h1 - elm-ae
# 	w1_in = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of h1
# 	w1_in, r = np.linalg.qr(w1_in)	# make orthogonal weight vectors
# 	h1 = hidden(training['signal'], w1_in)		# output of h1
# 	w1_out = np.matmul(np.linalg.pinv(h1), training['signal'])	# tune output weight of h1
# 	training['signal'] = np.matmul(h1, w1_out)	# the input signal after autoencoder

# 	# third hidden layer h2 - elm
# 	w2_in = np.random.normal(size = (training['signal'].shape[1], 3100))	# randomize input weights of h2
# 	h2 = hidden(training['signal'], w2_in)		# output of h2
# 	# w2_out = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]		# use least square to find the optimal output weight 
# 	w2_out = np.matmul(np.linalg.pinv(h2), training['thickness'])	# tune output weight for h2
# 	end = time.time()
# 	print('training time ', n_nodes/n, ': ', end - start)

# 	# start testing
# 	start = time.time()

# 	# h0
# 	h0 = hidden(testing['signal'], w0_in)	# output of h0
# 	testing['signal'] = np.matmul(h0, w0_out)		# after autoencoder

# 	# h1
# 	h1 = hidden(testing['signal'], w1_in)	# output of h0
# 	testing['signal'] = np.matmul(h1, w1_out)		# after autoencoder

# 	# h2
# 	h2 = hidden(testing['signal'], w2_in)
# 	approx = np.matmul(h2, w2_out)
# 	error = np.insert(error, error.shape[1], mse(approx, testing['thickness']), axis = 1)	# mean square error
# 	end = time.time()
# 	print('testing time: ', end - start)
# endto = time.time()
# print('total: ', endto - startto)
# # plot the mse 
# sio.savemat('error.mat', {'error': error})
# print('min mse index: ', np.argmin(error))
# plt.plot(np.linspace(100, n_nodes, n_nodes/n), error[0, :])
# plt.xlabel('Number of nodes')
# plt.ylabel('Mean square error')
# plt.show()
#################################################################################################