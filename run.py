import concurrent
import random

from CorrectEmAlgorithm import CorrectEmAlgorithm
from ArticleEmAlgorithm import ArticleEmAlgorithm
from truncatedGaussianMixture import TruncatedGaussianMixture
from constants import StoppingCriteria
from utils import setup_logging, save_output_to_pickle
import logging
import numpy as np


def run_simulation(y):
    setup_logging(y)
    seed = 592
    mean, variance, s, t, pweight, k, n, dim = y
    pokus = TruncatedGaussianMixture(mean, variance, s, t, pweight, k, n, dim, seed)
    data, weights = pokus.simulate()
    true_params = {'mean': mean, 'variance': variance, 'weights': weights}
    random.seed(seed)
    conc_data = np.concatenate(([data[j] for j in range(k)]))
    algo = CorrectEmAlgorithm(s, t, conc_data, dim, k, StoppingCriteria.loglikelihood_diff)
    algo_article = ArticleEmAlgorithm(s, t, conc_data, dim, k, StoppingCriteria.loglikelihood_diff)
    vysl, time = algo.em_algo()
    vysl_article, time_article = algo_article.em_algo()

    logging.info(f'Summary True parameters: \n {true_params}')
    logging.info(f'Summary Correct_EM_algorithm: \n {vysl[len(vysl) - 1]}')
    logging.info(f'Summary Article_EM_algorithm: \n {vysl_article[len(vysl_article) - 1]}')
    results = {'real_params': {'mean': mean, 'variance': variance, 's': s, 't': t, 'pweight': pweight, 'k': k, 'n': n, 'weight': weights}, 'data': data, 'correct_EM': vysl, 'article_EM': vysl_article, 'opti': algo.opti,
               'time': time, 'time_article': time_article}
    save_output_to_pickle(results, str(y)+'_seed_'+str(seed))

    return results


if __name__ == "__main__":
    params = [
        [[-8], [25], 0, 40, [1], 1, 150, 1]
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(run_simulation, params)
