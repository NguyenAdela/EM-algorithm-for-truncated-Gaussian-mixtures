import os
import logging
import random
import time

import pandas as pd

from CorrectEmAlgorithm import CorrectEmAlgorithm
from ArticleEmAlgorithm import ArticleEmAlgorithm

from constants import StoppingCriteria
from maindef import LOG_DIR
from utils import save_output_to_pickle


def setup_logging():
    random.seed(1993)
    current_time = time.strftime('%Y%m%d-%H%M%S')
    log_filepath = os.path.join(LOG_DIR, f'em_{current_time}_real_data')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        handlers=[
                            logging.FileHandler(log_filepath),
                            logging.StreamHandler()
                        ])


data = pd.read_csv("redwood.csv", header=0 ).values
data = data + [0,1]
seeds = [167, 224, 149, 273, 2812, 84]
setup_logging()
for s in seeds:
    algo = CorrectEmAlgorithm([0, 0], [1, 1], data, 2, 4, StoppingCriteria.loglikelihood_diff, seed=s)
    algo_article = ArticleEmAlgorithm([0, 0], [1, 1], data, 2, 4, StoppingCriteria.loglikelihood_diff, seed=s)
    vysl, time = algo.em_algo()
    vysl_article, time_article = algo_article.em_algo()

    logging.info(f'Summary Correct_EM_algorithm: \n {vysl[len(vysl) - 1]}')
    logging.info(f'Summary Article_EM_algorithm: \n {vysl_article[len(vysl_article) - 1]}')
    results = {'correct_EM': vysl, 'article_EM': vysl_article,
           'opti': algo.opti,'time': time, 'time_article': time_article}

    save_output_to_pickle(results, 'real_dataset_' + str(s))
