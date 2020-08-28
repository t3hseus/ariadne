# ariadne


# Setup repo environment:

1. Install miniconda
2. Choose the needed environment, with (environment_gpu.yml) or without CUDA (environment_cpu.yml)
3. Run 
```
conda env update --file environment_cpu.yml --name ariadne_cpu
conda activate ariadne_cpu
```

### note: to delete the conda environment run 
```
conda remove --name %NAME% --all
```
