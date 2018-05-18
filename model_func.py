import numpy as np 
import scipy.io as sio
import h5py as hp 
from dim_reduction import svd_reduction

def normalize_data(data):
# normalize data to put all the values in the range from 0 to 1
	return data/np.linalg.norm(data)

def bias(data):
# add bias to the first colomn of the data
	return np.insert(data, 0, 1, axis=1)

def import_data(filename, signal, thickness):
# load the data for training in the form of dictionary 
# filename (str): name of file to be loaded
# signal (str): key of the signal array in the dictionary
# thickness (str): key of the thickness
	data = hp.File(filename)	
	signal = np.array(data[signal]).transpose()['real']		# only take the real part of the signal as input 
	thickness = np.array(data[thickness])
	signal = normalize_data(signal)	# normalize the data
	signal, principal_dir = svd_reduction(signal)
	signal = bias(signal)	# add a bias column to the input signal 
	# thickness = normalize_data(thickness)
	return signal, thickness.transpose(), principal_dir

def split_set(signal, thickness):
	pos = round(len(signal) * 0.7)	# divide 70% of data as training set and the rest are for testing 
	training = {'signal': signal[0:pos, :], 'thickness': thickness[0:pos, :]}	# construct the training set, signal is the input data, thickness is the expoected output
	testing = {'signal': signal[pos:, :], 'thickness': thickness[pos:, :]}	# construct the tetsting set
	return training, testing