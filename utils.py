import logging
import pickle
import time
import os

from maindef import LOG_DIR


def setup_logging(params):
    current_time = time.strftime('%Y%m%d-%H%M%S')
    log_filepath = os.path.join(LOG_DIR, f'em_{current_time}_{str(params)}')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        handlers=[
                            logging.FileHandler(log_filepath),
                            logging.StreamHandler()
                        ])


def save_output_to_pickle(results, name):
    name = 'vysl_' + name.replace('.','-')
    with open(('results/'+name + ".pkl"), "wb") as f:
        pickle.dump(results, f)
