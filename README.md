# Numba and CUDA

To build CUDA code using NUMBA and explore other features.

For docker support requires NVIDIA container support [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and [Cuda on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

```
# Build the container
docker build --pull --rm -f "Dockerfile" -t numba101:latest "." 
# Launch container with gpu support.
>> docker run --rm -it --gpus all numba101:latest
```

On Windows:
```
# PowerShell
>> virtualenv .venv
>> .\.venv\Scripts\activate.ps1
>> pip install --upgrade numba
```