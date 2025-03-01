#!/bin/bash

#SBATCH --job-name=Cycles
#SBATCH --array=0-200
#SBATCH -c 2
#SBATCH -t 0-06:00
#SBATCH --mem-per-cpu=16gb
#SBATCH -o HOME-DIR/Subspace/lc_results_v4/fp-%j-%a.out

ml anaconda3-2019.03
conda activate neuro-2
python run_limit_cycle_dmft.py $SLURM_ARRAY_TASK_ID
