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







# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import time
# import sklearn.ensemble as ske
# from model_func import *

# # filename = 'data_10000_50_vf_8layers.mat'
# filename = 'data_10000_100_vf_8layers.mat'
# # filename = 'data_10000_100_8layers.mat'
# # filename = 'data_10000_75_8layers.mat'
# # filename = 'data_10000_50_8layers.mat'

# print('read data')
# start = time.time()
# training, testing, m, n = import_raw(filename)
# # training['thickness'] = training['thickness'] * n + m
# # testing['thickness'] = testign['thickness'] * n + m 
# end = time.time()
# print('prepare time: ', end - start)

# print('start training')
# start = time.time()
# # random mapping 
# # rm, r = np.linalg.qr(np.random.normal(size = (training['signal'].shape[1], 5000)))	# randomize input weights of h0
# # training['signal'] = np.matmul(training['signal'], rm)
# rfr = ske.RandomForestRegressor(n_estimators = 100, max_features = 1000, min_samples_split = 5, n_jobs = 6)
# rfr.fit(training['signal'], training['thickness'].transpose()[0])
# # h  = np.array([rfr.predict(training['signal'])]).transpose()
# # w_out = np.linalg.lstsq(h, training['thickness'], rcond = None)[0]
# end = time.time()
# print('training time: ', end - start)

# print('start testing')
# start = time.time()
# # testing['signal'] = np.matmul(testing['signal'], rm)
# approx = np.array([rfr.predict(testing['signal'])]).transpose()		# approximated values
# # approx = np.matmul(approx, w_out)
# error = mse(approx, testing['thickness'])
# print('mse: ', error)
# end = time.time()
# print('testing time: ', end - start)

# # save the network for future use 
# # sio.savemat('network.mat', {'w_in': w_in, 'w_out': w_out, 'mean': m, 'n': n})

# print('plot')
# approx = approx * n + m
# expected = testing['thickness'] * n + m
# plot_model(approx, testing['thickness'], n_points = testing['thickness'].shape[0])











import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import sklearn.neural_network as sknn
from model_func import *

filename = 'data_10000_100_vcyo_8layers.mat'

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
# random mapping 
# rm, r = np.linalg.qr(np.random.normal(size = (training['signal'].shape[1], 5000)))	# randomize input weights of h0
# training['signal'] = np.matmul(training['signal'], rm)
# construct multi-layer perceptron
mlnn = sknn.MLPRegressor(hidden_layer_sizes = (3000), activation = 'logistic', solver = 'adam', alpha = 0.01, batch_size = 3000, learning_rate_init = 0.001, max_iter = 1000, tol = 1e-5, verbose = True, shuffle = False)
# train
mlnn = mlnn.fit(training['signal'], training['thickness'].transpose()[0])
end = time.time()
print('training time: ', end - start)
print(mlnn.score(training['signal'], training['thickness'].transpose()[0]))

print('start testing')
start = time.time()
# testing['signal'] = np.matmul(testing['signal'], rm)
approx = np.array([mlnn.predict(testing['signal'])]).transpose()		# approximated values
# approx = np.matmul(approx, w_out)
error = mse(approx, testing['thickness'])
print('mse: ', error)
end = time.time()
print('testing time: ', end - start)

# save the network for future use 
# sio.savemat('network.mat', {'w_in': w_in, 'w_out': w_out, 'mean': m, 'n': n})

print('plot')
approx = approx * n + m
expected = testing['thickness'] * n + m
plot_model(approx, expected, n_points = testing['thickness'].shape[0])

print(mlnn.score(testing['signal'], testing['thickness'].transpose()[0]))