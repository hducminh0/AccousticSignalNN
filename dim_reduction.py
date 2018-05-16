import numpy as np
import numpy.linalg
def pca_reduction(data, n_components):
# function to perform dimensionality reduction using Principal Components Analysis 
# data: the data that needs to be reduced
# n_components: number of principal components we want to use
	data_adj = data - np.mean(data, axis=0)		# adjust the data to the center
	data_cov = np.cov(data_adj, rowvar=False)	# compute the covariance matrix of the adjusted data 
	eig_val, eig_vec = np.linalg.eig(data_cov)	# compute eigenvalues and eigenvectors of the covariance matrix
	sorted_eind = eig_val.ravel().argsort()[::-1]	# the indices of the sorted array of the eigenvalues
	chosen_comps = eig_vec[:, sorted_eind[0:n_components]]	# select the vectors with the highest eigenvalues to be the principal components
	reduced_data = np.matmul(chosen_comps.transpose(), data_adj.transpose()).transpose()	# project the data into a new dimension
	recovered_data = np.matmul(chosen_comps, reduced_data).transpose()	# recover the data from pca
	evaluation = np.sum(eig_val[sorted_eind[0:n_components]].transpose()) / np.sum(eig_val.transpose())		# evaluate how much data could we represent 
	return reduced_data, evaluation

def svd_reduction(data):
	# perform dimensionality reduction using Singular Value Decomposition 
	# we aim to retain 95% of the input data
	# data: input dataset that needs to be reduced
	u, s, v = np.linalg.svd(data)		# decomposes data matrix in to 3 components (X = U*S*V')
	goal = np.sum(np.square(s))	* 0.95	# we aim to retain 95% of the total data
	check = 0	# store the temporary sum of the singular value 
	n_s = 0		# number of singular values that we choose 
	# find the right number of singular values that we need to have 95% of the data
	while goal - check > 0:		
		check += s[n_s] ** 2
		n_s += 1
	# after the while loop, the value of n_s is the correct number of values
	reduced_data = np.matmul(u[:, 0:n_s], np.diag(s)[0:n_s, 0:n_s])		# reduce the data dimension
	recovered_data = np.matmul(reduced_data, v[0:n_s, :])	# recover the data 
	error_rate = np.sum(np.square(data - recovered_data)) / np.sum(np.square(data))		# calculate the error rate between the original data and the recovered one
	return reduced_data, n_s, error_rate