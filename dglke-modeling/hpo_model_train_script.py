import subprocess
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import re, os

# Define the search space
space = [
    Categorical([0.01, 0.05, 0.1, 0.25], name='lr'),
    Categorical([100, 250, 400], name='hidden_dim'),
    Categorical([2e-6, 2e-8, 1e-9], name='rc'),
    Categorical([9, 12, 19.9, 24], name='gamma'),
    Categorical([30, 64, 128], name='neg_sample_size')
]

DATA_PATH = '../data'
SAVE_PATH = './output'
DATASET = 'npkg'
FORMAT = 'raw_udd_hrt'
LOG_INTERVAL = 100
LOG_FILE_PATH = SAVE_PATH+'/training_hpo_log.log'

# Define the objective function
@use_named_args(space)
def objective(**params):
    model_name = 'TransR'
    log_file_path = os.path.join(SAVE_PATH, f'training_hpo_log_{params["lr"]}_{params["hidden_dim"]}_{params["rc"]}_{params["gamma"]}_{params["neg_sample_size"]}.log')

    # Construct the command with the current hyperparameters
    command = (
        f"DGLBACKEND=pytorch dglke_train --model_name {model_name} --data_path {DATA_PATH} --dataset {DATASET} --format {FORMAT} "
        f"--data_files train.csv valid.csv test.csv --delimiter , --save_path {SAVE_PATH} --no_save_emb --batch_size 512 --batch_size_eval 32 "
        f"--max_step 2500 --neg_sample_size {params['neg_sample_size']} --no_eval_filter --hidden_dim {params['hidden_dim']} --lr {params['lr']} -rc {params['rc']} "
        f"-log {LOG_INTERVAL} --gpu 0 1 2 3 --mix_cpu_gpu -adv -a 1.0 --valid --eval_interval 2500 --async_update --gamma {params['gamma']} --neg_sample_size_eval 500"
    )
    
    with open(log_file_path, 'w') as log_file:
        process = subprocess.run(command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Read the log file content
    with open(log_file_path, 'r') as log_file:
        output = log_file.read()
    
    # Extract validation MRR from the output
    mrr_pattern = re.compile(r"Valid average MRR: ([\d.]+)")
    mrr_values = [float(match.group(1)) for match in mrr_pattern.finditer(output)]

    if mrr_values:
        avg_mrr = sum(mrr_values)/len(mrr_values)
        return -avg_mrr
    
    return 0.0

# Run the optimization
res = gp_minimize(objective, space, n_calls=20, random_state=42)

# Print the best hyperparameters and corresponding MRR
print("Best hyperparameters: ", res.x)
print("Best validation MRR: ", -res.fun)
