#!/bin/bash

#SBATCH --job-name=heisenberg_j205
#SBATCH --output=sbatch_heisenberg_j205_out.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=2-
#SBATCH --mem-per-cpu=20G

module load miniconda CUDAcore/11.2.2 cuDNN/8.1.1.33-CUDA-11.2.2
conda deactivate
conda activate many_body
python notebooks/heisenberg_j205.py
# mprof run --python python notebooks/heisenberg_j205.py