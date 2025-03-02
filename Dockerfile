FROM nvidia/cuda:12.4.0-base-ubuntu22.04
LABEL Name=numba101 Version=0.1.0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# hadolint ignore=DL3008
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mercurial \
    subversion \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py312_24.9.2-0
ARG CONDA_SHA256=8d936ba600300e08eca3d874dee88c61c6f39303597b2b66baee54af4f7b4122

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_SHA256}  miniconda.sh" > miniconda.sha256 && \
    if ! sha256sum --status -c miniconda.sha256; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.sha256 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    # echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .

WORKDIR /app
COPY . /app

# Create a conda environment with name 'conda_test' and activate it by default
RUN conda create --name conda_test python=3.12 -y && \
    echo "conda activate conda_test" >> ~/.bashrc && \
    conda init bash

# SHELL ["/bin/bash", "--login", "-c"]

# RUN conda install --file requirements.txt
RUN conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
