#!/bin/bash -e
#SBATCH --job-name=pytorch_demo
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --time=00:01:00
#SBATCH --mem=512MB

module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/venv/bin/activate
srun python torch_demo.py

