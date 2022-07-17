from utils import *


class LedoitAndWolf_2001:
    """
    Implementation of Ledoit and Wolf (2001)
    "Improved Estimation of the Covariance Matrix of Stock Returns With an Application to Portfolio Selection"
    http://www.ledoit.net/ole2_abstract.htm
    """
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.num_variables = data_matrix.shape[1]
        self.num_observations = data_matrix.shape[0]
        self.data_matrix_centered = centering_data_matrix(self.data_matrix)

        self.MLE_estimator = sample_caovariance_matrix(self.data_matrix_centered)
        self.SIM_covariance = self.create_single_index_covariance_matrix()

    def compute_betas_and_x0t(self):
        betas = []
        x0t = np.sum(self.data_matrix_centered, axis=1)
        for i in range(0, self.num_variables):
            temp = cov_of_two_vec(self.data_matrix_centered[:, i], x0t)
            betas.append(temp)
        return x0t, betas

    def create_single_index_covariance_matrix(self):
        x0t, betas = self.compute_betas_and_x0t()
        var_x0t = var_of_vector(x0t)
        F = (1/var_x0t)*np.dot(np.matrix(betas).T, np.matrix(betas))
        for i in range(self.num_variables):
            F[i,i] = self.MLE_estimator[i, i]
        return F

    def compute_optimal_shrinkage_constant_for_SIM(self):

        # c
        c = compute_frobenius_norm(self.MLE_estimator, self.SIM_covariance)

        # p
        p_temp = np.zeros((self.num_variables, self.num_variables))
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                temp = 0.0
                for t in range(self.num_observations):
                    temp += \
                        np.power((self.data_matrix_centered[t, i]*self.data_matrix_centered[t, j] - self.MLE_estimator[i, j]), 2)
                p_temp[i, j] = (1.0/self.num_observations)*temp
        p = np.sum(p_temp)

        # r
        r_temp = np.zeros((self.num_variables, self.num_variables))
        x_0_t = np.sum(self.data_matrix_centered, axis=1)
        m_0 = np.mean(np.sum(self.data_matrix_centered, axis=1))

        s_i_0 = np.zeros((self.num_variables))
        for i in range(self.num_variables):
            s_i_0[i] = cov_of_two_vec(self.data_matrix_centered[:, i], x_0_t)
        s_00 = var_of_vector(x_0_t)
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                if i == j:
                    r_temp[i,j] = p_temp[i,j]
                    temp_2 = 0.0
                    for t in range(self.num_observations):
                        temp_2 +=\
                            (s_i_0[j]*s_00*self.data_matrix_centered[t, j] + s_i_0[i]*s_00*self.data_matrix_centered[t, i] - s_i_0[i]*s_i_0[j]*(x_0_t[t] - m_0)*(x_0_t[t] - m_0)*(self.data_matrix_centered[t, i])*(self.data_matrix_centered[t, j]))/(s_00*s_00) - self.SIM_covariance[i,j]* self.MLE_estimator[i,j]
                    r_temp[i, j] = (1.0 / self.num_observations) * temp_2
                else:
                    r_temp[i, j] = p_temp[i, j]
        r = np.sum(r_temp)

        shrinkage_param = (p - r) / c

        return shrinkage_param, p, r, c

    def compute_weighted_estimator(self):
        """
        The main function run it in order to get the weighted estimator
        """
        shrinkage_constant, p, r, c = self.compute_optimal_shrinkage_constant_for_SIM()
        shrinkage_parameter = shrinkage_constant/self.num_observations
        if 0 < shrinkage_parameter < 1:
            res = shrinkage_parameter*self.SIM_covariance + (1-shrinkage_parameter)*self.MLE_estimator
            # print 'using SIM'
        else:
            res = self.MLE_estimator
            # print 'not using SIM'
        return res, shrinkage_constant, shrinkage_parameter, p, r, c


