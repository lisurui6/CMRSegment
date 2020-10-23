FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y \
    vim tmux

RUN pip install --upgrade pip setuptools && \
    pip install nibabel vedo scikit-image tqdm pyhocon

RUN python setup.py install
