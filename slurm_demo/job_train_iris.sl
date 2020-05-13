#!/bin/bash -e
#SBATCH --job-name=train_iris
#SBATCH --time=00:01:00
#SBATCH --mem=1G

module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/venv/bin/activate
srun python ./aad/cmd/train.py -d Iris -e 200 -vlw

