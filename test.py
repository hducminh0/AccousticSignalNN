import numpy as np 
import scipy.io as sio 
from model_func import import_data, split_set
filename = 'data_10.mat'
signal = 'data/acc_signal'
thickness = 'data/thickness'
print('start read')
signal, thickness, principal_dir = import_data(filename, signal, thickness)
training, testing = split_set(signal, thickness)
sio.savemat('test.mat', training)
print('principal_dir')
print(principal_dir.shape)