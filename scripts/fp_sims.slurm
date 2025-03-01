#!/bin/bash

#SBATCH --job-name=FixedPoints
#SBATCH -c 4
#SBATCH --time 24:00:00
#SBATCH --mem-per-cpu=64gb
#SBATCH -o HOME-DIR/Subspace/fp_results_v5/fp-%j-%a.out
#SBATCH --gres=gpu:a40:1

ml anaconda3-2019.03
ml cuda/11.6.2
conda activate neuro-2
python run_fp_sims.py