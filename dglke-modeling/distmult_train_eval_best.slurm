#!/bin/bash
#SBATCH --job-name=distmult_dglke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --output=distmult_20240724.log
#SBATCH --error=distmult_20240724.err

#load env
module load python/ondemand-jupyter-python3.8

#activate environment
source activate /ix/rboyce/sbt12/envs/kg_embed_dgl

start_time=$(date +%s)

#DGLBACKEND=pytorch dglke_train --model_name DistMult --data_path ../data --dataset npkg --format raw_udd_hrt --data_files train.csv valid.csv test.csv \
#--delimiter , --save_path ./output --batch_size 512 --batch_size_eval 32 --max_step 2500 \
#--neg_sample_size 128 --no_eval_filter --hidden_dim 400 --lr 0.1 -rc 2e-8 -log 100 \
#--no_save_emb --gpu 0 1 2 3 --mix_cpu_gpu -adv -a 1.0 --neg_sample_size_eval 500 --gamma 24 --test --async_update

DGLBACKEND=pytorch dglke_train --model_name DistMult --data_path ../data --dataset npkg --format raw_udd_hrt --data_files train_with_weights.csv valid_with_weights.csv test_with_weights.csv \
--delimiter , --save_path ./output --batch_size 512 --batch_size_eval 32 --max_step 2500 \
--neg_sample_size 128 --no_eval_filter --hidden_dim 400 --lr 0.1 -rc 2e-8 -log 100 \
--no_save_emb --gpu 0 1 2 3 --mix_cpu_gpu -adv -a 1.0 --neg_sample_size_eval 500 --gamma 24 --test --async_update --has_edge_importance

end_time=$(date +%s)
duration=$((end_time-start_time))

echo "Job duration: $duration seconds" | tee -a distmult_20240724.log
crc-job-stats
