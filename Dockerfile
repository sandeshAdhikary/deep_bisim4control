FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu18.04

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    vim \
    # Rendering dependencies
    libglfw3 \
    libglew2.0 \
    libgl1-mesa-glx \
    libosmesa6 \
    # General dependencies
    wget \
    # Testing dependencies
    x11-apps \
    xpra \
    git

# Set up working directory
WORKDIR $HOME
COPY src/ src/
COPY conda_env.yaml .
    
# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
        /bin/bash Miniconda.sh -b -p /opt/conda && \
        rm Miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Create a conda environment
RUN conda env create -f conda_env.yaml -v
# Activate the conda env
RUN echo "source activate dbc" > ~/.bashrc