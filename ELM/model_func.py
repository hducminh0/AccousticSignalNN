import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py as hp

def normalize(data):
# normalize the data:
	m = np.mean(data)	# mean of the input data 
	n = max(data) - min(data)	
	nl_data = (data -  m)/n 	# mean normalization 
	return nl_data, m, n 	# return m and n so when we save the network we can get the actual value from the normalized one

def bias(data):
# add bias terms to the first column of the data
	return np.insert(data, 0, 1, axis=1)

def split_set(signal, thickness):
# split the input data into training and testing set
	pos = round(len(signal) * 0.75)	# divide 70% of data as training set and the rest are for testing 
	training = {'signal': signal[0:pos, :], 'thickness': thickness[0:pos, :]}	# construct the training set, signal is the input data, thickness is the expected output
	testing = {'signal': signal[pos:, :], 'thickness': thickness[pos:, :]}	# construct the tetsting set
	return training, testing

def open_file(filename, signal, thickness):
# open the .mat file to extract the signal and thickness
# filename: name of file to be loaded
# signal: key of the signal array in the dictionary
# thickness: key of the thickness
	data = hp.File(filename)	
	signal = np.array(data[signal]).transpose()		# only take the real part of the signal as input 
	thickness = np.array(data[thickness]).transpose()
	return signal, thickness

def mse(approx, expected):
# mean square error
	return np.sum((approx - expected) ** 2, axis=0)/(len(expected))		# mse to see how well the model performs 
	
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def hidden(signal, input_w, function = 'sigmoid'):
# hidden node in the hidden layer of the neural network 
# input_w: input weight of the hidden layer 
	if function == 'sigmoid':
		return sigmoid(np.matmul(signal, input_w))

def kernel(signal, d = 1,  function = 'rbf'):
# kernmel trick 
	k = np.zeros([signal.shape[0], signal.shape[0]])
	for i in range(0, signal.shape[0]):
		for j in  range(0, signal.shape[0]):
			if function == 'rbf':
				k[i, j] = rbf(signal[i, :], signal[j, :], d)
	return k

def rbf(a, b, d):
	return np.exp(-d * np.linalg.norm(a - b) ** 2)

def import_raw(filename, signal = 'data/acc_signal', thickness = 'data/thickness'):
# load the raw data for dimension reduction and training
# filename: name of file to be loaded
# signal: key of the signal array in the dictionary
# thickness: key of the thickness
	signal, thickness = open_file(filename, signal, thickness)
	thickness, m, n = normalize(thickness)
	training, testing = split_set(signal, thickness)
	testing['signal'] = bias(testing['signal'])	
	training['signal'] = bias(training['signal'])	# add a bias column to the input signal
	return training, testing, m, n

def plot_model(approx, expected, n_points = 100):
# plot the first n_points points of approximated and expeted results 
	expected = plt.scatter(np.linspace(0, n_points, n_points), expected[0:n_points], color = 'red')
	approx, = plt.plot(np.linspace(0, n_points, n_points), approx[0:n_points], color = 'black')
	plt.legend([expected, approx], ['Expected output', 'Approximated model'])
	plt.xlabel('Signal index')
	plt.ylabel('Thickness')
	plt.show()
