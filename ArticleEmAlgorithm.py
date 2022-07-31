import logging

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from numpy.linalg import cholesky
from EmAlgorithm import EmAlgorithm


class ArticleEmAlgorithm(EmAlgorithm):
    def logpdf_norm(self, x, mean, variance):
        """Logarithm of normal pdf"""
        if self.dim == 1:
            return norm.logpdf(x, mean, variance ** 0.5)
        else:
            return multivariate_normal.logpdf(x, mean, variance)

    def cdf_norm(self, u, mean, variance):
        """Cdf of normal distribution"""
        if self.dim == 1:
            return norm.cdf(u, mean, variance ** 0.5)
        else:
            return multivariate_normal.cdf(u, mean, variance)

    def M_step(self):
        """Maximization step from the article"""
        if self.dim == 1:
            self.weight = np.empty((1, self.k))
            tmu = {}
            tvar = {}
            for j in range(self.k):
                alpha = (self.t - self.mean[j]) / self.variance[j] ** 0.5
                beta = (self.s - self.mean[j]) / self.variance[j] ** 0.5
                tmu[j] = self.variance[j] ** 0.5 * (
                        norm.pdf(alpha) - norm.pdf(beta)) / (
                                 norm.cdf(beta) - norm.cdf(alpha))

                tvar[j] = self.variance[j] * (
                        1 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / (norm.cdf(beta) - norm.cdf(alpha))
                        - ((norm.pdf(alpha) - norm.pdf(beta)) / (norm.cdf(beta) - norm.cdf(alpha))
                           ) ** 2)

            self.weight = np.array([(1 / self.n) * np.sum(self.z[0:self.n, j]) for j in range(self.k)])
            self.mean = np.array(
                [(np.dot(self.z[0:self.n, j], self.x) / np.sum(self.z[0:self.n, j])) - tmu[j] for j in
                 range(self.k)]).reshape((self.k,))
            variance = np.array([np.sum([self.z[i, j] * (self.x[i] - self.mean[j]) * (
                (self.x[i] - self.mean[j])) for i in range(self.n)], axis=0) / np.sum(
                self.z[0:self.n, j]) + self.variance[j] - (tvar[j] + tmu[j] * tmu[j])
                                 for j in range(self.k)]).reshape((self.k,))
            self.variance = variance
        else:
            self.weight = np.empty((1, self.k))
            tmu = {}
            tvar = {}
            for j in range(self.k):
                self.r.assign('mu', self.mean[j] - self.mean[j])
                self.r.assign('sigma', self.variance[j])
                self.r.assign('s', self.s - self.mean[j])
                self.r.assign('t', self.t - self.mean[j])
                self.r('''
                         mmnts <- mtmvnorm(
                                    mean = mu,
                                    sigma = sigma,
                                    lower = s,
                                    upper = t
                                )
                        ''')

                tmu[j] = self.r.get('mmnts$tmean')
                tvar[j] = self.r.get('mmnts$tvar')

            tvar = np.array(list(tvar.values()))
            tmu = np.array(list(tmu.values()))
            self.weight = np.array([(1 / self.n) * np.sum(self.z[0:self.n, j]) for j in range(self.k)])
            z_sum = self.z.sum(axis=0)
            self.mean = (self.z[..., None] * self.x.squeeze()[:, None, :]).sum(axis=0) / z_sum[..., None] - tmu
            # Centered x
            xc = self.x.squeeze()[:, None, :] - self.mean

            # "H_k"
            h = self.variance - (tvar + (tmu[..., None] @ tmu[:, None, :]))

            variance = np.sum((xc[..., None] @ xc[..., None, :]) * self.z[..., None, None], axis=0) / z_sum[:, None, None] + h

            self.variance = variance
            for k in range(self.k):
                try:
                    cholesky(self.variance[k])
                except np.linalg.LinAlgError:
                    self.variance[k] = self.variance[k] +1e-4*np.identity(self.dim)
                    try:
                        cholesky(self.variance[k])
                    except np.linalg.LinAlgError:
                        self.variance[k] = self.variance[k] +1e-3*np.identity(self.dim)
                        try:
                            cholesky(self.variance[k])
                        except np.linalg.LinAlgError:
                            self.variance[k] +1e-2*np.identity(self.dim)
                    logging.error('Matrix Sigma ', str(k), ' is not positive definite.')

            for k in range(self.k):
                try:
                    multivariate_normal.cdf(self.s, self.mean[k], self.variance[k])
                except ValueError:
                    self.variance[k] = self.variance[k] +1e-4*np.identity(self.dim)
                    try:
                        multivariate_normal.cdf(self.s, self.mean[k], self.variance[k])
                    except ValueError:
                        self.variance[k] = self.variance[k] +1e-3*np.identity(self.dim)
                        try:
                            multivariate_normal.cdf(self.s, self.mean[k], self.variance[k])
                        except ValueError:
                            self.variance[k] +1e-2*np.identity(self.dim)

        if self.dim > 1:
            for k in range(self.k):
                try:
                    cholesky(self.variance[k])
                except np.linalg.LinAlgError:
                    logging.error('Matrix Sigma ', str(k), ' is not positive definite.')
        else:
            for k in range(self.k):
                if self.variance[k] <= 0:
                    logging.error('Variance ', str(k), ' is not positive.')

        self.params[self.nstep] = {'mean': self.mean, 'variance': self.variance, 'weight': self.weight,
                                   'log-likelihood': self.log_likelihood_calculation()}
