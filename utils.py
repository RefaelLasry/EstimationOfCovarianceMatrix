import numpy as np


def centering_data_matrix(data_matrix):
    means_of_variables = np.mean(data_matrix, axis=0)
    centered_data_matrix = \
        data_matrix - means_of_variables*np.ones((data_matrix.shape[0], data_matrix.shape[1]))
    return centered_data_matrix


def take_a_sample(data_matrix, sample_size):
    sample = data_matrix[
             np.random.choice(data_matrix.shape[0], int(data_matrix.shape[0]*sample_size),
                              replace=False), :]
    # print 'sample taken with dim of ', sample.shape[0], ' variables and ', sample.shape[1], 'observations'
    return sample


def sample_caovariance_matrix(data_matrix):
    return (1.0/data_matrix.shape[0])*data_matrix.T.dot(data_matrix)


def compute_frobenius_norm(population_matrix, estimator_matrix):
    return np.linalg.norm((population_matrix - estimator_matrix), 'fro')


def cov_of_two_vec(x, y):
    cov = np.sum(np.multiply((x - np.mean(x)), (y-np.mean(y))))/len(x)
    return cov


def var_of_vector(x):
    return np.sum(np.power(x - np.mean(x), 2))/len(x)
