import numpy as np


class DirectKernel:
    """
    Implementation of the article:
    "Direct Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices"
    Ledoit and Wolf, Oct 2017,
    translated from authors' Matlab code
    """
    def __init__(self, X):
        self.X = X
        self.n = None
        self.p = None
        self.sample = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.L = None
        self.h = None

    def pav(self, y):
        """
        PAV uses the pair adjacent violators method to produce a monotonic
        smoothing of y
        translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
        """
        y = np.asarray(y)
        assert y.ndim == 1
        n_samples = len(y)
        v = y.copy()
        lvls = np.arange(n_samples)
        lvlsets = np.c_[lvls, lvls]
        flag = 1
        while flag:
            deriv = np.diff(v)
            if np.all(deriv >= 0):
                break

            viol = np.where(deriv < 0)[0]
            start = lvlsets[viol[0], 0]
            last = lvlsets[viol[0] + 1, 1]
            s = 0
            n = last - start + 1
            for i in range(start, last + 1):
                s += v[i]

            val = s / n
            for i in range(start, last + 1):
                v[i] = val
                lvlsets[i, 0] = start
                lvlsets[i, 1] = last
        return v

    def estimate_cov_matrix(self):

        # extract sample eigenvalues sorted in ascending order and eigenvectors
        self.n, self.p = self.X.shape
        self.sample = (self.X.transpose() @ self.X) / self.n
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.sample)
        isort = np.argsort(self.eigenvalues, axis=-1)
        self.eigenvalues.sort()
        self.eigenvectors = self.eigenvectors[:, isort]

        # compute direct kernel estimator
        self.eigenvalues = self.eigenvalues[max(1, self.p - self.n + 1) - 1:self.p]
        self.L = np.repeat(self.eigenvalues, min(self.n, self.p), axis=0).reshape(self.eigenvalues.shape[0], min(self.n, self.p))
        self.h = self.n ** (-0.35)
        component_00 = 4*(self.L.T**2)*self.h**2 - (self.L - self.L.T)**2
        component_0 = np.maximum(np.zeros((component_00.shape[1], component_00.shape[1])), component_00)
        component_a = np.sqrt(component_0)
        component_b = 2*np.pi*(self.L.T**2)*self.h**2
        ftilda = np.mean(component_a / component_b, axis=1)

        com_1 = np.sign(self.L - self.L.T)
        com_2_1 = (self.L - self.L.T)**2 - 4*self.L.T**2*self.h**2
        com_2 = np.maximum(np.zeros((com_2_1.shape[1], com_2_1.shape[1])), com_2_1)
        com_3_1 = np.sqrt(com_2)
        com_3_2 = com_1 * com_3_1
        com_3 = com_3_2 - self.L + self.L.T
        com_4 = 2*np.pi*self.L.T**2*self.h**2
        com_5 = com_3 / com_4
        Hftilda = np.mean(com_5, axis=1)

        if self.p <= self.n:
            com_0 = (np.pi*(self.p/self.n)*self.eigenvalues*ftilda)**2
            com_1 = (1 - (self.p / self.n) - np.pi * (self.p / self.n) * self.eigenvalues * Hftilda) ** 2
            com_2 = com_0 + com_1
            dtilde = self.eigenvalues / com_2
        else:
            Hftilda0 = (1-np.sqrt(max(1-4*self.h**2, 0))) / (2*np.pi*self.n*self.h**2)*np.mean(1/self.eigenvalues)
            dtilde0 = 1/(np.pi*((self.p-self.n)/self.n)*Hftilda0)
            dtilde1 = self.eigenvalues/np.pi**2*self.eigenvalues**2*(ftilda**2+Hftilda**2)
            dtilde = np.hstack((dtilde0*np.ones((self.p-self.n, 1)).reshape(self.p-self.n,), dtilde1))

        dhat = self.pav(dtilde)
        sigmahat = np.dot(self.eigenvectors, (np.tile(dhat, (self.p, 1)).T * self.eigenvectors.T))
        return sigmahat


if __name__ == '__main__':
    # X = np.array([[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]])
    # X = np.array([[3, 4, 5], [7, 8, 9], [11, 12, 13]])
    # X = np.random.rand(700, 480)

    matrixSize = 420
    A = np.random.rand(matrixSize, matrixSize)
    B = np.dot(A, A.transpose())
    cov = B
    print('real cov matrix :')
    print(cov)

    num_samples = 760
    mean = np.zeros(matrixSize)
    X = np.random.multivariate_normal(mean, cov, num_samples)

    X = X - X.mean(axis=0) * np.ones((num_samples, matrixSize))

    sample_matrix = X.T@X / float(num_samples)
    print('sample_matrix :')
    print(sample_matrix)

    dk = DirectKernel(X)
    em = dk.estimate_cov_matrix()
    print('direct kernel :')
    print(em)

