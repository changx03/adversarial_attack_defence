#!/bin/bash -e
#SBATCH --job-name=CifarResNet
#SBATCH --output=slurm_log/%x_%j.out
#SBATCH --time=20:00:00
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-30

module load CUDA
module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/venv/bin/activate

srun $(head -n $SLURM_ARRAY_TASK_ID jobs/script_CifarResNet.txt | tail -n 1)
