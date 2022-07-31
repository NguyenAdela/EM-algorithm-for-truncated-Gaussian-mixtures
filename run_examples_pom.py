import concurrent

from examples import *
from simulation import run_simulation

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(run_simulation, params_1d_2c_overlap_3)
