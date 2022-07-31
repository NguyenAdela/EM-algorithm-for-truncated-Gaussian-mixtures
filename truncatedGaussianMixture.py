import numpy as np
from pyper import R
from scipy.stats import norm
from scipy.stats import multivariate_normal
import logging

from maindef import R_PATH


class TruncatedGaussianMixture:

    def __init__(self, mean, variance, s, t, p, k, n, dim, seed=167):
        self.weights = None
        self.dim = dim
        self.mean = np.array(mean)
        self.variance = np.array(variance)
        self.s = s
        self.t = t
        self.p = np.array(p)
        self.k = k
        self.n = n
        self.dim = dim
        self.x = dict()
        self.seed = seed

    def calculate_weights(self):
        if self.k == 1:
            self.weights = [1]
        else:
            if self.dim == 1:
                self.weights = self.p * (
                        norm.cdf(self.t, self.mean, self.variance ** 0.5)
                        - norm.cdf(self.s, self.mean, self.variance ** 0.5)
                ).reshape(self.k,) / np.dot(
                    self.p,
                    (
                            norm.cdf(self.t, self.mean, self.variance ** 0.5)
                            - norm.cdf(self.s, self.mean, self.variance ** 0.5)
                    ))
            else:
                self.weights = self.p * (
                    [multivariate_normal.cdf(self.t, mu, sigma)
                     - multivariate_normal.cdf(self.s, mu, sigma) for mu, sigma in zip(self.mean, self.variance)]
                )

    def simulate(self):
        """Simulate a truncated Gaussian mixture"""
        self.calculate_weights()
        logging.info('##################################################################################')
        logging.info('##################################################################################')
        logging.info(f"##### Generating truncated Gaussian mixture using {self.__class__.__name__} #####")
        logging.info('##################################################################################')
        logging.info('##################################################################################\n')
        logging.info(f'Number of clusters: {self.k}\n')
        logging.info(f"True distribution: \nmean: \n{self.mean} \nvariance: \n{self.variance} \nweights: {self.weights} \nbounds: {self.s, self.t}")

        self.calculate_weights()
        npoints = np.random.multinomial(self.n, self.weights)
        logging.info(f"Performing simulation of Gaussian mixture..\n")
        for j in range(self.k):
            if npoints[j] > 0:
                r = R(R_PATH, use_pandas=True)
                r.assign('n', npoints[j])
                r.assign('mean', self.mean[j])
                r.assign('variance', self.variance[j])
                r.assign('s', self.s)
                r.assign('t', self.t)
                r.assign('seed', self.seed)
                r('''
                    library(tmvtnorm)
                    set.seed(seed)
                    x <- rtmvnorm(n, mean, variance, lower = s, upper=t)
                    ''')
                self.x[j] = r.get('x')
            else:
                self.x[j] = np.array([])

        return self.x, self.weights

    def simulate_2(self):
        x = dict()
        self.calculate_weights()
        for k in range(self.k):
            x[k] = []
        if self.dim == 1:
            for i in range(self.n):
                z_i = np.argmax(np.random.multinomial(1, self.p))
                x_i = np.random.normal(self.mean[z_i], self.variance[z_i] ** 0.5)
                if self.s <= x_i <= self.t:
                    x[z_i].append([x_i])
        else:
            for i in range(self.n):
                z_i = np.argmax(np.random.multinomial(1, self.p))
                x_i = np.random.multivariate_normal(self.mean[z_i], self.variance[z_i])
                if np.all([self.s[i] <= x_i[i] <= self.t[i] for i in range(self.dim)]):
                    x[z_i].append([x_i])

        return x, self.weights

    def pdf(self, x):
        """Array of truncated Gaussian pdfs for all elements of the mixture

        Returns: (n x k) matrix with (i, j)-th element being the value of the pdf for truncated Gaussian pdf
        for i-th observation with parameters self.mean[j], self.variance[j]"""
        if self.dim == 1:
            # Reshape for correct broadcasting in norm.pdf
            mu = self.mean.reshape(-1, 1)
            variance = self.variance.reshape(-1, 1)

            result = norm.pdf(x, mu, variance ** 0.5) / (
                    norm.cdf(self.t, mu, variance ** 0.5) - norm.cdf(self.s, mu, variance ** 0.5))
        else:

            result = np.array([multivariate_normal.pdf(x, mean, variance) / (
                    multivariate_normal.cdf(self.t, mean, variance) - multivariate_normal.cdf(self.s, mean, variance)) for
                               mean, variance in zip(self.mean, self.variance)])

        return result.transpose()
