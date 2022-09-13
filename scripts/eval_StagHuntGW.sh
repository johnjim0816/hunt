#!/bin/sh
env="StagHuntGW"
seed_max=3

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=5 python eval/eval.py --env_name ${env} --seed ${seed} --num_agents 2 --episode_length 50 --model_dir "./results/StagHuntGW/paper-new1/" --eval_episodes 3 --save_gifs
done
