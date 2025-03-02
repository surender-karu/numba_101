# Numba and CUDA

To build CUDA code using NUMBA and explore other features.

For docker support NVIDIA container toolkit is required [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and [Cuda on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

```
# Build the container
>> docker build --pull --rm -f "Dockerfile" -t numba101:latest "." 
# Launch container with gpu support.
>> docker run --rm -it --gpus all numba101:latest
```

On Windows to run locally (requires a GPU with CUDA support):\
> Install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) \
> Create a virtual environment using conda (make sure path is updated)
```
# PowerShell
>> conda create -n numba_env
# Conda init to initialize conda with powershell for first use.
>> conda init
>> conda activate numba_env
# Install packages from requirements.txt
>> conda install --file .\requirements.txt
# Validate installation
>> python.exe .\matrix.py

```