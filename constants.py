import enum


class StoppingCriteria(enum.Enum):
    loglikelihood_diff = 1
    sigma              = 2
    mu                 = 3
    weight             = 4
    all_params         = 5
    n_step             = 6
