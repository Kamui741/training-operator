FROM ubuntu:20.04

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装 NVIDIA 驱动程序
RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.39/NVIDIA-Linux-x86_64-460.39.run -O /tmp/nvidia-driver.run \
    && chmod +x /tmp/nvidia-driver.run \
    && /tmp/nvidia-driver.run --silent \
    && rm /tmp/nvidia-driver.run

# 安装 CUDA Toolkit（可选）
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get -y install cuda
