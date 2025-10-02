#!/bin/bash

#SBATCH --job-name=heisenberg
#SBATCH --output=sbatch_heisenberg-%j.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2-
#SBATCH --mem-per-cpu=16G
#SBATCH --constraint doubleprecision

module load miniconda CUDA/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1
conda activate many_body
python notebooks/heisenberg.py