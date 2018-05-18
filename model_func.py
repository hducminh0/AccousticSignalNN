import numpy as np 
import scipy.io as sio

def normalize_data(data):
# normalize data to put all the values in the range from 0 to 1
	return data/np.linalg.norm(data)

def bias(data):
# add bias to the first colomn of the data
	return np.insert(data, 0, 1, axis=1)

def read_data(filename, signal, thickness, principal_dir):
# load the data for training in the form of dictionary 
# filename: name of file to be loaded
# signal: key of the signal array in the dictionary
# thickness: key of the thickness
# principal_dir: key of the principal direction 
	data = sio.loadmat(filename)	
	signal = data[signal]
	signal = bias(signal)	# add a bias column to the input signal 
	signal = normalize_data(signal.real)	# normalize the data
	thickness = data[thickness]
	# thickness = normalize_data(thickness)
	pos = round(len(signal) * 0.7)	# divide 70% of data as training set and the rest are for testing 
	training = {'signal': signal[0:pos, :], 'thickness': thickness[0:pos, :]}	# construct the training set, signal is the input data, thickness is the expoected output
	testing = {'signal': signal[pos:, :], 'thickness': thickness[pos:, :]}	# construct the tetsting set
	principal_dir = data[principal_dir].real
	return training, testing, principal_dir	