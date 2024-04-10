#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=hist-eq
#SBATCH --gpus=1
#SBATCH --output=hist-eq.log

FILE=src/sample

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm $FILE.cu -o $FILE
srun  $FILE valve.png valve_out.png