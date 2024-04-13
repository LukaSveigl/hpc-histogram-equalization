#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=hist-eq
#SBATCH --gpus=1
#SBATCH --output=hist-eq.log

#FILE=src/hist-eq-parallel
FILE=src/hist-eq-parallel-up

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm $FILE.cu -o $FILE
srun  $FILE data/input/7680x4320.png data/output/par/7680x4320.png