import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from model_func import *

# filename = 'data_10_100.mat'
# filename = 'data_5000.mat'
filename = 'data_10000_50_vf.mat'

print('read data')
start = time.time()
training, testing, m, n = import_raw(filename)	# import data 
end = time.time()
print('prepare time: ', end - start)

n_nodes = 2100
print('start training')
start = time.time()
np.random.seed(seed = None)
w_i0 = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize weight btwn input layer and first hidden layer h0 
# put 2 hidden layers as 1
h0 = hidden(training['signal'], w_i0)	# output of the combined hidden layer
w_out = np.matmul(np.linalg.pinv(h0), training['thickness'])	# output weight of the network 
# separate 2 layers 
h1 = hidden(training['thickness'], np.linalg.pinv(w_out))
w_01 = np.matmul(np.linalg.pinv(h0), hidden_inv(h1))		# weight btwn layer h0 and h1 
h1 = hidden(h0, w_01)	# actual output of h1
# h1 = hidden(w_01, h0)	# actual output of h1
w_out = np.matmul(np.linalg.pinv(h1), training['thickness'])

end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
h0 = hidden(testing['signal'], w_i0)
h1 = hidden(h0, w_01)
approx = np.matmul(h1, w_out)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start) 

# save the network for future use 
sio.savemat('network.mat', {'w_i0': w_i0, 'w_01': w_01, 'w_out': w_out, 'mean': m, 'n': n})

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