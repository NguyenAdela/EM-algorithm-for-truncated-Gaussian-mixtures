import os
import random

from pyper import R

from abc import abstractmethod, ABC
import logging
import time

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from sklearn.cluster import KMeans
from numpy.linalg import cholesky
from dotenv import load_dotenv

load_dotenv()

R_PATH = os.getenv('R_PATH')


class EmAlgorithm(ABC):

    def __init__(self, s, t, x, dim, k, stopping_criteria_selection, case='clasic', seed=23):

        self.s = s
        self.t = t
        self.n = len(x)
        self.x = x
        self.dim = dim
        self.stopping_criteria_selection = stopping_criteria_selection
        self.r = R(R_PATH, use_pandas=True)
        self.r('''
            library(tmvtnorm)
            set.seed(1993) 
            ''')
        self.case = case
        self.k = k
        self.z = None
        self.init = None
        self.mean = None
        self.variance = None
        self.weight = None
        self.q = None
        self.seed = seed

        self.opti = dict()
        self.params = dict()
        self.nstep = 0

    def n_clusters(self):
        """Estimate the number of clusters.

        Currently hard-coded."""
        self.k = self.k

    def q2v(self, q):
        return q[np.tril_indices(self.dim)]

    def v2q(self, v):
        q = np.zeros((self.dim, self.dim))
        q[np.tril_indices(self.dim)] = v
        return q

    @staticmethod
    def moments_univariate_tg(mean, variance, s, t):

        alpha = (s - mean) / variance ** 0.5
        beta = (t - mean) / variance ** 0.5
        z = norm.cdf(beta) - norm.cdf(alpha)
        tmu = mean + variance ** 0.5 * (norm.pdf(alpha) - norm.pdf(beta)) / z

        tvar = variance * (1 + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / z
                           - ((norm.pdf(alpha) - norm.pdf(beta)) / z) ** 2)

        return tmu, tvar

    def log_likelihood_calculation(self):
        if self.dim == 1:
            lf = np.empty((self.n, self.k))
            l_sum = np.empty(self.n)
            for n in range(self.n):
                for k in range(self.k):
                    lf[n, k] = (1 / (
                                norm.cdf(self.t, self.mean[k], self.variance[k] ** 0.5) - norm.cdf(self.s, self.mean[k],
                                                                                                   self.variance[
                                                                                                       k] ** 0.5))) * (
                                           self.weight[k] * norm.pdf(self.x[n], self.mean[k], self.variance[k] ** 0.5))
                l_sum[n] = np.log(np.sum([lf[n, k] for k in range(self.k)]))
            l_final = np.sum([l_sum[n] for n in range(self.n)])
        else:
            lf = np.empty((self.n, self.k))
            l_sum = np.empty(self.n)
            for n in range(self.n):
                for k in range(self.k):
                    lf[n, k] = (1 / (multivariate_normal.cdf(self.t, self.mean[k],
                                                             self.variance[k]) - multivariate_normal.cdf(self.s,
                                                                                                         self.mean[k],
                                                                                                         self.variance[
                                                                                                             k]))) * (
                                           self.weight[k] * multivariate_normal.pdf(self.x[n], self.mean[k],
                                                                                    self.variance[k]))
                l_sum[n] = np.log(np.sum([lf[n, k] for k in range(self.k)]))
            l_final = np.sum([l_sum[n] for n in range(self.n)])
        return l_final

    def initial_distribution(self):
        """Compute initial distribution using K-means"""
        kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=500, n_init=25, random_state=self.seed)
        kmeans.fit(self.x.reshape(-1, self.dim))
        if self.dim == 1:
            self.mean = np.array(kmeans.cluster_centers_).reshape(self.k, )
        else:
            self.mean = np.array(kmeans.cluster_centers_).reshape(self.k, self.dim)
        counts = np.unique(kmeans.labels_, return_counts=True)[1]
        data_sep = [self.x[kmeans.labels_ == j] for j in range(self.k)]

        # Use sample variance for variance
        if self.case == 'homoskedastic':
            variance = np.cov(self.x.T, ddof=1)
            self.variance = [variance for _ in range(self.k)]
        elif self.case == 'isotropic':
            self.variance = np.array(
                [np.var(np.concatenate([data_sep[j][i] for i in range(self.dim)])) * np.identity(self.dim) for j in
                 range(self.k)])
        else:
            self.variance = np.array([np.cov(data_sep[j].reshape((-1, self.dim)).T, ddof=1) for j in range(self.k)])
        self.weight = counts / self.n
        self.weight = self.weight.reshape(1, self.k)

        if self.dim == 1:
            if self.case == 'homoskedastic':
                self.init = np.concatenate((self.mean.flatten(), self.variance[0].flatten()))
            else:
                self.init = np.concatenate((self.mean.flatten(), self.variance.flatten()))
        else:
            print([self.variance[j] for j in range(self.k)])
            v = np.array([self.q2v(cholesky(self.variance[j])) for j in range(self.k)])
            if self.case == 'homoskedastic':
                self.init = np.concatenate((self.mean.flatten(), v[0].flatten()))
            elif self.case == 'isotropic':
                self.init = np.concatenate(self.mean.flatten(), np.array(
                    [np.sqrt(np.var(np.concatenate([data_sep[j][i] for i in range(self.dim)]))) for j in
                     range(self.k)]).flatten())
            else:
                self.init = np.concatenate((self.mean.flatten(), v.flatten()))

    def E_step(self):
        """Expectation step. Estimates z"""

        if self.dim == 1:
            pdf = np.empty((self.n, self.k))
            self.z = np.empty((self.n, self.k))

            for k in range(self.k):
                pdf[:, k] = norm.pdf(self.x.squeeze(), self.mean[k], self.variance[k] ** 0.5) / (norm.cdf(self.t, self.mean[k], self.variance[k] ** 0.5) - norm.cdf(self.s, self.mean[k], self.variance[k] ** 0.5))
        else:
            pdf = np.empty((self.n, self.k))
            self.z = np.empty((self.n, self.k))
            for k in range(self.k):
                pdf[:, k] = multivariate_normal.pdf(self.x.squeeze(), self.mean[k], self.variance[k]) / (multivariate_normal.cdf(self.t, self.mean[k], self.variance[k]) - multivariate_normal.cdf(self.s, self.mean[k], self.variance[k]))
        self.weight = self.weight.reshape(1, self.k)
        for n in range(self.n):
            for k in range(self.k):
                self.z[n, k] = self.weight[0, k] * pdf[n, k] / np.sum(
                    [pdf[n, m] * self.weight[0, m] for m in range(self.k)])

    @abstractmethod
    def M_step(self):
        """Maximization step. Estimates w, mean, variance"""
        pass

    def em_algo(self, tolerance=10 ** -6):

        start_time = time.time()
        logging.info('##################################################################################')
        logging.info('##################################################################################')
        logging.info(f"#####                EM clustering using {self.__class__.__name__}                 ####")
        logging.info('##################################################################################')
        logging.info('##################################################################################\n')
        logging.info('Determining number of clusters...')
        self.n_clusters()
        logging.info(f'Number of clusters: {self.k}')
        logging.info(f'Calculating initial distribution...')
        self.initial_distribution()
        lf = np.empty((self.n, self.k))
        l_sum = np.empty(self.n)
        if self.dim == 1:
            for n in range(self.n):
                for k in range(self.k):
                    lf[n, k] = (1 / (norm.cdf(self.t, self.mean[k], self.variance[k] ** 0.5) - norm.cdf(self.s,
                                                                                                        self.mean[
                                                                                                            k],
                                                                                                        self.variance[
                                                                                                            k] ** 0.5))) * (
                                       self.weight[0, k] * norm.pdf(self.x[n],
                                                                    self.mean[k],
                                                                    self.variance[k] ** 0.5))
                l_sum[n] = np.log(np.sum([lf[n, k] for k in range(self.k)]))
            l_final = np.sum([l_sum[n] for n in range(self.n)])
        else:
            for n in range(self.n):
                for k in range(self.k):
                    lf[n, k] = (1 / (
                            multivariate_normal.cdf(self.t, self.mean[k], self.variance[k]) - multivariate_normal.cdf(self.s, self.mean[k], self.variance[k]))) * (
                                       self.weight[0, k] * multivariate_normal.pdf(self.x[n],
                                                                                   self.mean[k],
                                                                                   self.variance[k]))
                l_sum[n] = np.log(np.sum([lf[n, k] for k in range(self.k)]))
            l_final = np.sum([l_sum[n] for n in range(self.n)])

        self.params[0] = {'mean': self.mean, 'variance': self.variance, 'weight': self.weight, 'log-likelihood': l_final}
        logging.info(
            f'Initial distribution:\nmean: \n{self.mean} \nvariance: \n{self.variance} \nweight: \n{self.weight}\n \nbounds: \n{self.s, self.t}\n')
        should_continue = True

        while should_continue:
            self.nstep += 1
            logging.info(f'### Step {self.nstep}')
            logging.info(f'Performing expectation step...')
            self.E_step()
            logging.info(f'Performing maximization step...')
            self.M_step()
            logging.info(
                f'Current parameters: \nmean:      \n{self.mean} \nvariance:   \n{self.variance} \nweights: \n{self.weight}')

            logging.info('Stopping criteria:')
            if self.nstep > 1:
                variance_current = self.params[self.nstep]['variance']
                variance_previous = self.params[self.nstep - 1]['variance']

                current_loglikelihood = self.params[self.nstep]['log-likelihood']
                logging.info(
                    f'Log-likelihood: \nlog-likelihood:     \n{current_loglikelihood}'
                )
                previous_loglikelihood = self.params[self.nstep - 1]['log-likelihood']
                difference_loglikelihood = np.abs(current_loglikelihood - previous_loglikelihood)
                should_continue = (difference_loglikelihood > tolerance)
                if self.nstep >= 10000:
                    should_continue = False
                logging.info(f'Current variance:      \n{variance_current}')
                logging.info(f'Previous variance:     \n{variance_previous}')
                # logging.info(f'Difference:         \n{difference}')
            else:
                should_continue = True

            logging.info('Step complete.\n')

        logging.info(f'Completed in {self.nstep} step{"s" if self.nstep > 1 else ""}. '
                     f'Final parameters: \nmean: \n{self.mean} \nvariance: \n{self.variance} \nweights: \n{self.weight}')

        end_time = time.time()
        logging.info(f'Calculation done in {end_time - start_time:.2f}s\n')
        return self.params, end_time - start_time
