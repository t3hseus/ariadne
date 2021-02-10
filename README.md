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
