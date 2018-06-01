import numpy as np
import matplotlib.pyplot as plt
from model_func import *

# filename = 'data_10.mat'
# filename = 'data_5000.mat'
filename = 'data_10000_100_vf.mat'

print('read data')
training, testing, m, n = import_raw(filename)
input_w, out_w1 = import_nn('network_10000_100_vf_4000nodes_nldata.mat')
h = hidden(testing['signal'], input_w)
approx = np.matmul(h, out_w1)


print('plot')
plot_model(approx, testing['thickness'])