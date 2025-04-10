#!/bin/bash
#SBATCH --job-name=train_grpo
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00  # Adjust as needed
#SBATCH --mem=32G         # Adjust memory as needed
#SBATCH --cpus-per-task=4 # Adjust CPU cores as needed

/scratch/michael/miniconda3/envs/trl/bin/accelerate launch --num_processes=2 train_grpo.py
