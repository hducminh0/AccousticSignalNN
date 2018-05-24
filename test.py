import numpy as np
import matplotlib.pyplot as plt
from model_func import *

# filename = 'data_10.mat'
# filename = 'data_5000.mat'
filename = 'data_10000_100.mat'

print('read data')
training, testing = import_raw(filename)
input_w, output_w = import_nn('network_10000_100.mat')
h = hidden(testing['signal'], input_w)
approx = np.matmul(h, output_w)

expected = plt.scatter(np.linspace(0, 99, 100), testing['thickness'][0:100], color = 'red')
approximated, = plt.plot(np.linspace(0, 99, 100), approx[0:100], color = 'black')
plt.legend([expected, approximated], ['Expected output', 'Approximated model'])
plt.xlabel('Signal index')
plt.ylabel('Thickness')
plt.show()