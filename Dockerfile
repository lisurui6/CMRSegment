FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY . /workspace/CMRSegment
WORKDIR /workspace/CMRSegment

RUN apt-get update && apt-get install -y build-essential git libjpeg-dev && \
    apt-get install -y vim tmux

RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt

RUN python setup.py install
