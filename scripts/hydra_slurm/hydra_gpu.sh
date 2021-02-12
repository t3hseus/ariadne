#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1

lscpu
nvidia-smi
conda env update --file environment_gpu.yml --name ariadne_gpu_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ariadne_gpu_new
"$@"