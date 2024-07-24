import re
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

log_file_path = 'distmult_20240723.log'
# Define the search space
space = [
    Categorical([0.01, 0.05, 0.1, 0.25], name='lr'),
    Categorical([100, 250, 400], name='hidden_dim'),
    Categorical([2e-6, 2e-8, 1e-9], name='regularization_coef'),
    Categorical([9, 12, 19.9, 24], name='gamma'),
    Categorical([30, 64, 128], name='neg_sample_size')
]

@use_named_args(space)
def objective_from_log(**params):
    with open(log_file_path, 'r') as log_file:
        output = log_file.read()

        mrr_pattern = re.compile(r"Valid average MRR: ([\d.]+)")
        mrr_values = [float(match.group(1)) for match in mrr_pattern.finditer(output)]

        if mrr_values:
            avg_mrr = sum(mrr_values)/len(mrr_values)
            return -avg_mrr
        
        return 0.0
    
if __name__ == "__main__":
    

    # Run the optimization using the log file
    res = gp_minimize(objective_from_log, space, n_calls=10, random_state=42)

    print('Best hyperparameters: ', res.x)
    print('Best validation MRR: ', -res.fun)