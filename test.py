import numpy as np 
import h5py as hp 
from dim_reduction import svd_reduction 

data = hp.File('data_5000.mat')
acc_signal = np.array(data['data_5000/acc_signal']).transpose()
acc_signal = acc_signal['real'] + acc_signal['imag'] * 1j
reduced_signal, n_retained_dim, error_rate = svd_reduction(acc_signal)
reduced_signal
n_retained_dim
error_rate