#!/bin/bash
#SBATCH -p knltut -t 3600

lscpu
conda env update --file environment_cpu.yml --name ariadne_cpu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ariadne_cpu
echo $PYTHONPATH
"$@"