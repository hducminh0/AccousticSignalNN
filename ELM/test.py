# import numpy as np
# import matplotlib.pyplot as plt
# from model_func import *

# # filename = 'data_10.mat'
# # filename = 'data_5000.mat'
# filename = 'data_10000_100_vf.mat'

# print('read data')
# training, testing, m, n = import_raw(filename)
# input_w, out_w1 = import_nn('network_10000_100_vf_4000nodes_nldata.mat')
# h = hidden(testing['signal'], input_w)
# approx = np.matmul(h, out_w1)


# print('plot')
# plot_model(approx, testing['thickness'])

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import sklearn.ensemble as ske
from model_func import *

# filename = 'data_10.mat'
# filename = 'data_5000.mat'
# filename = 'data_10000_50_vf.mat'
filename = 'data_10000_50_vf_8layers.mat'

print('read data')
start = time.time()
training, testing, m, n = import_raw(filename)
training['thickness'] = training['thickness'] * n + m
testing['thickness'] = testing['thickness'] * n + m 
end = time.time()
print('prepare time: ', end - start)

print('start training')
start = time.time()
# random mapping 
# rm, r = np.linalg.qr(np.random.normal(size = (training['signal'].shape[1], 5000)))	# randomize input weights of h0
# training['signal'] = np.matmul(training['signal'], rm)
rfr = ske.RandomForestRegressor(n_estimators = 100, max_features = 10000, min_samples_split = 10, n_jobs = 6)
rfr.fit(training['signal'], training['thickness'].transpose()[0])
# h  = np.array([rfr.predict(training['signal'])]).transpose()
# w_out = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]
end = time.time()
print('training time: ', end - start)

print('start testing')
start = time.time()
# testing['signal'] = np.matmul(testing['signal'], rm)
approx = np.array([rfr.predict(testing['signal'])]).transpose()		# approximated values
# approx = np.matmul(approx, w_out)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
# sio.savemat('network.mat', {'w_in': w_in, 'w_out': w_out, 'mean': m, 'n': n})

print('plot')
# approx = approx * n + m
# expected = testing['thickness'] * n + m
plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import time
# from model_func import *

# # filename = 'data_10.mat'
# # filename = 'data_5000.mat'
# # filename = 'data_10000_50_vf.mat'
# filename = 'data_10000_50_vf_8layers.mat'
# # filename = 'data_1000_100_vf_8layers.mat'

# print('read data')
# start = time.time()
# training, testing, m, n = import_raw(filename)
# end = time.time()
# print('prepare time: ', end - start)

# # # random mapping 
# # rm, r = np.linalg.qr(np.random.normal(size = (training['signal'].shape[1], 1000)))	# randomize input weights of h0
# # training['signal'] = np.matmul(training['signal'], rm)

# n_est = 1900
# n_nodes = 10	# number of node in the hidden layer
# n_samples = 1000
# np.random.seed(seed = None)
# w_in = np.empty([n_est, training['signal'].shape[1], n_nodes])
# w_out = np.empty([n_nodes, n_est])


# print('start training')
# start = time.time()
# for i in range(0, n_est):
# 	ind = np.round(np.random.rand(1, n_samples) * training['signal'].shape[0] - 1)[0].astype('int')
# 	w_in[i, :, :] = np.random.normal(size = (training['signal'].shape[1], n_nodes))	# randomize input weights of the network 
# 	h = hidden(training['signal'][ind, :], w_in[i, :, :])		# output of the hidden layer
# 	w_out[:, i] = np.linalg.lstsq(h, training['thickness'][ind], rcond = None)[0][:, 0]		# use least square to find the optimal output weight 
# 	# w_out = np.matmul(np.linalg.pinv(h), training['thickness'])
# end = time.time()
# print('training time: ', end - start)

# print('start testing')
# start = time.time()
# # testing['signal'] = np.matmul(testing['signal'], rm)
# temp = np.empty([testing['thickness'].shape[0], n_est])
# for i in range(0, n_est):
# 	h = hidden(testing['signal'], w_in[i, :, :])	# output of the hidden layer
# 	temp[:, i] = np.matmul(h, w_out[:, i])		# approximated values
# approx = np.array([np.mean(temp, axis = 1)]).transpose()
# error = mse(approx, testing['thickness'])
# print('mse: ', error)
# end = time.time()
# print('testing time: ', end - start)

# # plot the first 100 samples 
# print('plot')
# approx = approx * n + m
# testing['thickness'] = testing['thickness'] * n + m
# plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])