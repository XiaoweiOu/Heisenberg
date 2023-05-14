#!/bin/bash

#SBATCH --job-name=sbatch_SUN_6
#SBATCH --output=sbatch_SUN_6_out.txt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-
#SBATCH --mem-per-cpu=20G

module load miniconda
conda deactivate
conda activate many_body
python notebooks/heisenberg_energy.py