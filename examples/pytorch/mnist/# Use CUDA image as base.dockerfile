# Use CUDA image as base
ARG CUDA_DOCKER_VERSION=11.3.1-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

# Arguments for version control
ARG TENSORFLOW_VERSION=2.9.2
ARG PYTORCH_VERSION=1.12.1+cu113
ARG TORCHVISION_VERSION=0.13.1+cu113
ARG CUDNN_VERSION=8.2.1.32-1+cuda11.3
ARG NCCL_VERSION=2.9.9-1+cuda11.3
ARG PYTHON_VERSION=3.8

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Install basic dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn8=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openssh-client \
        openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-4.1.4-bin.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz && \
    cd /tmp && tar -zxf /tmp/openmpi-4.1.4-bin.tar.gz && \
    mkdir openmpi-4.1.4/build && cd openmpi-4.1.4/build && ../configure --prefix=/usr/local && \
    make -j all && make install && ldconfig && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Create Python symlink and install pip
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow and PyTorch with Horovod
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html

RUN pip install --no-cache-dir tensorflow==${TENSORFLOW_VERSION}

# Install Horovod
RUN pip install --no-cache-dir future typing packaging
RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install horovod

# Check Horovod installation
RUN python -c "import horovod.tensorflow as hvd; hvd.init()" && \
    python -c "import horovod.torch as hvd; hvd.init()"
