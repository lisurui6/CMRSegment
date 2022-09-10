FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

RUN apt-key del 7fa2af80 \
    && rm /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub \
    && apt-get update && apt-get install -y build-essential git libjpeg-dev \
    && apt-get install -y vim tmux ffmpeg libsm6 libxext6

RUN pip install --upgrade pip setuptools && \pip install nibabel scikit-image tqdm pyhocon pandas matplotlib opencv-python \
    && pip install SimpleITK trimesh meshcut tensorboard boundary-curvature shapely rasterio open3d vtk==9.0.1 vedo

RUN pip install torch==1.12.1

RUN cd /workspace/ && \
    git clone https://github.com/lisurui6/neural_renderer.git && \
    cd ./neural_renderer && python setup.py install

RUN cd /workspace/ && git clone https://github.com/lisurui6/CCITK.git && cd ./CCITk && python setup.py develop
