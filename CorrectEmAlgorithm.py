import logging

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

import scipy.optimize as optimize

from EmAlgorithm import EmAlgorithm


class CorrectEmAlgorithm(EmAlgorithm):

    def logpdf_norm(self, x, mu, sigma):
        """Logarithm of normal pdf"""
        if self.dim == 1:
            return norm.logpdf(x, mu, sigma ** 0.5)
        else:
            return multivariate_normal.logpdf(x, mu, sigma, allow_singular=True)

    def cdf_norm(self, u, mu, sigma):
        """Cdf of normal distribution"""
        if self.dim == 1:
            return norm.cdf(u, mu, sigma ** 0.5)
        else:
            return multivariate_normal.cdf(u, mu, sigma, allow_singular=True)

    def log_function(self, y):
        """Log-likelihood function"""

        if self.dim == 1:
            if self.case == 'homoskedastic':
                mu, sigma = np.split(y, (self.k,))
                sigma = np.array([sigma for _ in range(self.k)]).flatten()
            elif self.case == 'isotropic':
                mu, sigma = np.split(y, (self.k,))
                sigma = np.array([np.identity * sigma[k] for k in range(self.k)])
            else:
                mu, sigma = np.split(y, 2)
            log_likelihood = -np.sum(self.z * (
                    np.log(self.weight) + self.logpdf_norm(self.x, mu, sigma) - np.log(self.cdf_norm(self.t, mu, sigma) - self.cdf_norm(self.s, mu, sigma))))
        else:
            mu, v = np.split(y, (self.k * self.dim,))

            if self.case == 'homoskedastic':
                q = np.array([self.v2q(v) for _ in range(self.k)])
                mu = np.split(mu, self.k)
                sigma = [np.dot(q, q.T) for _ in range(self.k)]
            elif self.case == 'isotropic':
                mu = np.split(mu, self.k)
                sigma = [v[k] ** 2 * np.identity(self.dim) for k in range(self.k)]
            else:
                mu = np.split(mu, self.k)
                q = np.array([self.v2q(np.array_split(v, self.k)[k]) for k in range(self.k)])
                q = q.reshape((self.k, self.dim, self.dim))
                sigma = [np.dot(q[j], q[j].T) for j in range(self.k)]
            log_likelihood = -np.sum(self.z * (
                    np.log(self.weight).reshape((self.k, 1))
                    + [
                        self.logpdf_norm(self.x, mu[i], sigma[i])
                        - np.log(
                            self.cdf_norm(self.t, mu[i], sigma[i]) - self.cdf_norm(self.s, mu[i], sigma[i])
                        )
                        for i in range(self.k)]
            ).T)

        return log_likelihood

    def M_step(self):
        """Maximization step. Estimates weight, mu, sigma"""

        self.weight = (1 / self.n) * self.z.sum(axis=0)
        if self.dim == 1:
            mle = optimize.minimize(self.log_function, x0=self.init, method='Nelder-Mead', options={'maxiter': 10000})
            self.opti[self.nstep] = mle
            if self.case == 'homoskedastic':
                self.mean, v = np.split(mle.x, (self.k * self.dim,))
                self.variance = [v for _ in range(self.k)]
                self.init = np.concatenate((self.mean, v))
            else:
                self.mean, self.variance = np.split(mle.x, (self.k * self.dim,))
                self.init = np.concatenate((self.mean, self.variance))
        else:
            mle = optimize.minimize(self.log_function, x0=self.init, method='Nelder-Mead', options={'maxiter': 100000})
            self.opti[self.nstep] = mle
            self.mean, v = np.split(mle.x, (self.k * self.dim,))
            self.mean = np.array(self.mean).reshape(self.k, self.dim)
            if self.case == 'homoskedastic':
                self.q = np.array([self.v2q(v) for _ in range(self.k)])
                self.variance = np.array([np.dot(self.q[j], self.q[j].T) for j in range(self.k)])
                self.init = np.concatenate((self.mean.flatten(), v))
            elif self.case == 'isotropic':
                self.variance = np.array([v[k] ** 2 * np.identity(self.dim) for k in range(self.k)])
            else:
                q = np.array([self.v2q(
                    np.array_split(v, self.k)[k]) for k in range(self.k)])
                self.variance = np.array([np.dot(q[j], q[j].T) for j in range(self.k)])

                self.init = np.concatenate((self.mean.flatten(), v))
        if mle.success:
            logging.info(f'Number of iterations: {mle.nit}, function value: {mle.fun}')
        else:
            logging.warning('Optimization was unsuccessful.')
            logging.warning(f'Optimization object: {mle}')

        self.params[self.nstep] = {'mean': self.mean, 'variance': self.variance, 'weight': self.weight,
                                   'complete_data_log-likelihood': -mle.fun,
                                   'log-likelihood': self.log_likelihood_calculation()}
