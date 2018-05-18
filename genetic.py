import numpy as np
import scipy.io as sio

def populate(n_gen, n_chrom):
# initilize the population for genetic algorithm
# n_gen: number of genes in a chromosomes (row)
# n_chrom: total number of chormosomes in the population (column)
	return np.random.rand(n_gen, n_chrom)

def crossover(population, mse):
# crossover the best parents to produce offsping
# population: population we generated 
# mse: mean square error, scores show how well each chromosome in the population performs.
	sorted_ind = np.argsort(mse)	# sort the error in ascending order 
	for i in range(round(population.shape[1] / 2)):
		r = np.random.rand()*0.02	# randomize a coefficient for crossover
		s = sorted_ind[0, -1-i]		# take the last position which is the bad chromosome 
		population[:, s] = r * population[:, sorted_ind[0, i]] + (1 - r) * population[:, sorted_ind[0, i + 1]]	# arithmetic crossover operator
		if np.random.rand() <= 0.05:	# chance that uniform mutation occurs 
			if np.random.rand() < 0.5:
				population[round(np.random.rand()), s] += np.random.rand()	
			else:
				population[round(np.random.rand()), s] -= np.random.rand()
	return population