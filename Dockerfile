FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Set timezone and environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV USER=user
ENV VENV=bevformer_distill
ENV PYTHON=3.8
ENV CONDA_DIR=/home/${USER}/miniconda
ENV PATH=$CONDA_DIR/envs/${VENV}/bin:$CONDA_DIR/bin:$PATH

# Switch to bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install basic libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    wget \
    vim \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA paths
RUN echo 'PATH=$PATH:/usr/local/cuda-11.1/bin' >> ~/.bashrc && \
    echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64' >> ~/.bashrc && \
    echo 'CUDA_HOME=/usr/local/cuda-11.1' >> ~/.bashrc

# Customize bash prompt
RUN echo "PS1='\[\033[0;32m\]\u@\h:\[\033[0;34m\]\w\[\033[0m\]\$ '" >> ~/.bashrc

# Create workspace directory
RUN mkdir -p /home/${USER}
WORKDIR /home/${USER}

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O /home/${USER}/miniconda.sh && \
    /bin/bash /home/${USER}/miniconda.sh -b -p $CONDA_DIR && \
    rm /home/${USER}/miniconda.sh

# Update Conda
RUN conda update -y conda && conda clean -a -y

# Copy environment.yaml and create the conda environment
RUN conda create -n ${VENV} python=${PYTHON} -y

# Add Conda activation to bashrc
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> ~/.bashrc

# Install additional pip packages
RUN pip install \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda install -c omgarcia gcc-6
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0 \
                mmsegmentation==0.14.1

# For mmdetection3D installation before "python setup.py install"
RUN pip install einops \
                fvcore \
                seaborn \
                iopath==0.1.9 \
                timm==0.6.13 \
                typing-extensions==4.5.0 \
                pylint \
                ipython==8.12 \
                numpy==1.19.5 \
                matplotlib==3.5.2 \
                numba==0.48.0 \
                pandas==1.4.4 \
                scikit-image==0.19.3 \
                setuptools==59.5.0 \
                scipy==1.10.1 \
                pywavelets==1.4.1 \
                scikit-learn==1.3.2 \
                anyio==3.3.0 \
                lyft_dataset_sdk \
                "networkx>=2.2,<2.3" \
                nuscenes-devkit==1.1.10 \
                plyfile \
                tensorboard \
                "trimesh>=2.35.39,<2.35.40"

# Detectron2 installation
RUN python -m pip install detectron2==0.5 -f \
                https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

RUN pip install pillow==9.5.0

# Default command
CMD ["/bin/bash"] 