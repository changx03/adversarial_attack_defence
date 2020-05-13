#!/bin/bash -e
#SBATCH --job-name=array_demo
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --time=00:01:00     
#SBATCH --mem=512MB
#SBATCH --cpus-per-task=1
#SBATCH --array=1-2

cat script_demo_array.sh | sed -n ${SLURM_ARRAY_TASK_ID}p 
