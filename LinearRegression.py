import numpy as np
import scipy.io as sio
from genetic import populate, crossover
from model_func import import_data, split_set

filename = 'data_10.mat'
signal = 'data/acc_signal'
thickness = 'data/thickness'
signal, thickness, principal_dir = import_data(filename, signal, thickness)
training, testing = split_set(signal, thickness)
w = populate(training['signal'].shape[1], training['signal'].shape[1] * 2)	# initialize the weight matrix as a population for genetic algorithm 
# w = populate(training['signal'].shape[1], 30)
mse = np.ones([1, w.shape[1]])	# set the mean square error to 1
i = 0
# start training the model
while (mse > 0.07).any():
	print(i)
	approx = np.matmul(training['signal'], w)	# compute the approximated output
	# mse = np.sqrt(np.sum((approx - training['thickness']) ** 2, axis=0)/(len(training['thickness'])))
	mse = np.sum((approx - training['thickness']) ** 2, axis=0)/(len(training['thickness']))	# mean square error
	w = crossover(w, np.array([mse]))	# run the crossover for genetic algorithm 
	i += 1
	print(mse)

approx = np.matmul(testing['signal'], w)	# test on the model
mse = np.sum((approx - testing['thickness']) ** 2, axis=0)/(len(training['thickness']))		# mse to see how well the model performs 
sio.savemat('LinearRegressionModel.mat', {'weight': w[:, np.argsort(mse)[0]]})
