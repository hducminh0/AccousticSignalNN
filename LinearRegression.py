import numpy as np
import scipy.io as sio
from genetic import populate, crossover
from model_func import read_data

filename = 'reduced_signal_10.mat'
signal = 'signal'
thickness = 'thickness'
principal_dir = 'principal_dir'
training, testing, principal_dir = read_data(filename, signal, thickness, principal_dir)
w = populate(training['signal'].shape[1], training['signal'].shape[1] * 2)	# initialize the weight matrix as a population for genetic algorithm 
# w = populate(training['signal'].shape[1], 30)
mse = np.ones([1, w.shape[1]])	# set the mean square error to 1
i = 0
# start training the model
while (mse > 0.07).any():
	print(i)
	approx = np.matmul(training['signal'], w)	# compute the approximated output
	# rmse = np.sqrt(np.sum((approx - training['thickness']) ** 2, axis=0)/(len(training['thickness'])))
	mse = np.sum((approx - training['thickness']) ** 2, axis=0)/(len(training['thickness']))	# mean square error
	w = crossover(w, np.array([mse]))	# run the crossover for genetic algorithm 
	i += 1

approx = np.matmul(testing['signal'], w)	# test on the model
mse = np.sum((approx - testing['thickness']) ** 2, axis=0)/(len(training['thickness']))		# mse to see how well the model performs 

sio.savemat('LinearRegressionModel.mat', {'weight': w[:, np.argsort(mse)[0]]})