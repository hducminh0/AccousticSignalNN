import numpy as np 
import h5py as hp 
import scipy.io as sio
from dim_reduction import svd_reduction 

# data = hp.File('data_5000.mat')
# acc_signal = np.array(data['data_5000/acc_signal']).transpose()
data = hp.File('data_10.mat')
acc_signal = np.array(data['data/acc_signal']).transpose()
acc_signal = acc_signal['real'] + acc_signal['imag'] * 1j
# acc_signal = acc_signal['real']
reduced_signal, principal_dir, error_rate = svd_reduction(acc_signal)
reduced_signal = {'signal': reduced_signal, 'principal_dir': principal_dir}
# sio.savemat('reduced_signal_5000.mat', reduced_signal)
sio.savemat('reduced_signal_10.mat', reduced_signal)
print('done')