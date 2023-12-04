#!/bin/bash

#SBATCH --job-name=hubbard
#SBATCH --output=sbatch_hubbard_out.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16G

module load miniconda CUDAcore/11.2.2 cuDNN/8.1.1.33-CUDA-11.2.2
conda deactivate
conda activate many_body
python notebooks/hubbard.py