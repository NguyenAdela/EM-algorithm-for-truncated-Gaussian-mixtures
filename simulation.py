import pickle
import random

from CorrectEmAlgorithm import CorrectEmAlgorithm
from ArticleEmAlgorithm import ArticleEmAlgorithm
from truncatedGaussianMixture import TruncatedGaussianMixture
from constants import StoppingCriteria
from twoDimensionalVisualization import SummaryPlot2D
from utils import setup_logging, save_output_to_pickle
import logging
import numpy as np


def run_simulation(y):
    setup_logging(y)
    random.seed(167)
    mean, variance, s, t, pweight, k, n, dim = y
    pokus = TruncatedGaussianMixture(mean, variance, s, t, pweight, k, n, dim)
    data, weights = pokus.simulate()
    true_params = {'mean': mean, 'variance': variance, 'weights': weights}

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
    save_output_to_pickle(results, str(y))

    #summary_plot = SummaryPlot2D(results)
    #summary_plot.output_plot()

    return results


'''
result = dict()

if __name__ == "__main__":
    params = [
        [[-4, 15], [25, 25], 0, 100, [0.5, 0.5], 2, 1000, 1]
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(run_simulation, params)


if __name__ == "__main__":
    s = 0
    t = 1
    weights = [0.5,0.5]
    m = np.arange(-0.1,0.6,0.1)
    sigma = np.arange(0.05, 0.2,0.05)
    results = {}
    params = []
    for mi in m:
        results[mi] = {}
        for sigmai in sigma:
            params.append([[mi, 1-mi], [sigmai,sigmai], s, t,  weights, 2, 1000, 1])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(run_simulation, params)


objects = []
params = [
    [[[5, 15]], [[[20, 0], [0, 5]]], [0, 0], [25, 25], [1], 1, 2000, 2],
    [[[10, 10]], [[[5, 0], [0, 20]]], [0, 0], [25, 25], [1], 1, 2000, 2]
]


for param in params:
    print(param)
    with (open('./vysl_'+str(param)+'.pkl', 'rb')) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    vysl=objects[-1]

    summary_plot = SummaryPlot2D(vysl)
    summary_plot.output_plot()
'''