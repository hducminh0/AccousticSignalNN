import numpy as np
import scipy.io as sio
import time
from model_func import *

# filename = 'data_10.mat'
filename = 'data_5000.mat'
# filename = 'data_10000.mat''

print('read data')
start = time.time()
training, testing, principal_dir = import_data(filename)
end = time.time()
print('prepare time: ', end - start)

# start training the model
print('start training')
start = time.time()
w = np.linalg.lstsq(training['signal'], training['thickness'])[0]	# training the linear model using least square method 
end = time.time()
print('train time: ', end - start)

# testing
print('start testing')
approx = np.matmul(testing['signal'], w)	# test on the model
print('approx: ', approx)
print('thickness ', testing['thickness'])
error = mse(approx, testing['thickness'])		# error to see how well the model performs 
print('error: ', error)
print('weight: ', w)
# print('error: ', error.shape)
# print(error)
# print('weight: ', w.shape)
sio.savemat('LinearRegressionModel.mat', {'weight': w, 'principal_dir': principal_dir})	# save the model