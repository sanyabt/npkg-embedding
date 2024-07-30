#!/bin/bash
#SBATCH --job-name=train_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --output=hpo_output.log
#SBATCH --error=hpo_output.err

#load env
module load python/ondemand-jupyter-python3.8

#activate environment
source activate /ix/rboyce/sbt12/envs/kg_embed_dgl/

start_time=$(date +%s)

python hpo_model_train_script.py

end_time=$(date +%s)
duration=$((end_time-start_time))

echo "Job duration: $duration seconds" | tee -a hpo_output.log
