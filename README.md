# ariadne


# Setup environment:

1. Install miniconda
2. Choose the needed environment, with (environment_gpu.yml) or without CUDA (environment_cpu.yml)
3. If you are installing with CUDA, check your drivers first ([here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html))
4. Run
```
conda env update --file environment_cpu.yml --name ariadne_cpu
conda activate ariadne_cpu
```

### note: to delete the conda environment run
```
conda remove --name %NAME% --all
```

# Training

To start training procedure execute `train.py` script and pass to it a path to the
training configuration file

```
python train.py --config resources/gin/tracknet_v2_train.cfg
```

Ariadne uses `logging`, so to specify the log level one should use `--log` parameter. E.g.:

```
python train.py --config resources/gin/tracknet_v2_train.cfg --log DEBUG
```

The default loggin level is `INFO`.


# Run scripts on HybriLIT

There are several utility scripts to facilitate ariadne execution on the `hydra` JINR cluster:

1. `scripts/hydra/hydra_cpu.sh`
2. `scripts/hydra/hydra_gpu.sh`
3. `scripts/hydra/govorun_gpu.sh`

The main syntax is:
```
sbatch $SCRIPT_PATH $command_to_be_executed
```

### Hydra

For example, to execute training script on the GPU queue of hydra cluster:
1. Verify that the miniconda has installed in the `~/miniconda3` or manually change the path in the 
script you want to execute. Row with `source ~/miniconda3/etc/profile.d/conda.sh` command

2. Run `scripts/hydra/hydra_gpu.sh` script
```
sbatch scripts/hydra/hydra_gpu.sh python train.py --config resources/gin/tracknet_v2_train.cfg
```

3. The `slurm-jobid.out` file with stdout will appear in the root directory.

### GOVORUN

Executing a command on GOVORUN differs from executing Hydra commands only in the need to add a module for working with the supercomputer.

```
module add GVR/v1.0-1 && \
sbatch scripts/hydra/govorun_gpu.sh python train.py --config resources/gin/tracknet_v2_train.cfg
```