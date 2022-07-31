import concurrent

import numpy as np

from examples import params_1d_1c, params_1d_2c, params_2d_1c, params_2d_2c
from simulation import run_simulation

if __name__ == "__main__":

    s = 0
    t = 1
    weights = [0.5,0.5]
    m = np.arange(-0.1,0.6,0.1)
    sigma = 0.5
    params_rel_bias = []
    for mi in m:
        params_rel_bias.append([[mi, 1-mi], [sigma,sigma], s, t,  weights, 2, 500, 1])

    for param in [params_rel_bias]:
        params = param
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(run_simulation, params)
