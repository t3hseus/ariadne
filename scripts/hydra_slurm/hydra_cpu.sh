#!/bin/bash
#SBATCH -p cpu

lscpu
nvidia-smi
conda env update --file environment_cpu.yml --name ariadne_cpu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ariadne_cpu
"$@"