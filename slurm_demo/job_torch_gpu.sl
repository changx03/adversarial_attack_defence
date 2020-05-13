#!/bin/bash -e
#SBATCH --job-name=torch_gpu
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

module load CUDA
module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/venv/bin/activate
srun python torch_gpu.py
