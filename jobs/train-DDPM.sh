#!/bin/bash
#SBATCH --job-name=train-DDPM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmartinez@hs.uni-hamburg.de
#SBATCH --output=/work/bbd0953/DDPM/jobs/output_%x.log
#SBATCH --export=NONE

# Activate venv
source /work/bbd0953/DDPM/.venv/bin/activate

# Run job:
cd /work/bbd0953/DDPM
# -u for unbuffered output
CUDA_VISIBLE_DEVICES=0 python -u -m model.train