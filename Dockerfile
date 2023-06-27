FROM ghcr.io/uwrobotlearning/mujoco_docker:latest

RUN apt-get update \
    && apt-get install -y \
    git

WORKDIR $HOME

# Set up the conda environment
COPY conda_env.yml .

# Create a conda environment
RUN conda env create -f conda_env.yml -v
# Activate the conda env
RUN echo "source activate dbc" > ~/.bashrc