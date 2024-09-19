#!/bin/bash
#SBATCH --job-name=transe_dglke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --output=transe_20240722.log
#SBATCH --error=transe_20240722.err

#load env
module load python/ondemand-jupyter-python3.8

#activate environment
source activate /ix/rboyce/sbt12/envs/kg_embed_dgl

start_time=$(date +%s)

DGLBACKEND=pytorch dglke_train --model_name TransE_l1 --data_path ../data --dataset npkg --format raw_udd_hrt --data_files train.csv valid.csv test.csv \
--delimiter , --save_path ../output --batch_size 512 --batch_size_eval 32 --max_step 2500 \
--eval_interval 500 --neg_sample_size 128 --no_eval_filter --hidden_dim 400 --lr 0.25 -rc 1e-9 -log 100 \
--gpu 0 1 2 3 --mix_cpu_gpu -adv -a 1.0 --neg_sample_size_eval 500 --valid --gamma 19.9 --test --async_update

end_time=$(date +%s)
duration=$((end_time-start_time))

echo "Job duration: $duration seconds" | tee -a transe_20240722.log
crc-job-stats
