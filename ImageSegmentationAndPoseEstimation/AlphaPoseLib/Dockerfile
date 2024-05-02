FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 as install_deps

RUN DEBIAN_FRONTEND=noninteractive apt update && DEBIAN_FRONTEND=noninteractive apt install -y --allow-change-held-packages \
    ffmpeg \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    git \
    libgl1 \
    libglib2.0-0\
    libsm6 \
    libxrender1 \
    libxext6 \
    curl 

RUN pip install pip --upgrade
RUN pip install torch torchvision torchaudio \
    cython \
    pycocotools \
    gdown \
    matplotlib \
    cython_bbox \
    scipy \
    opencv-python \
    pyyaml \
    tensorboardx \
    terminaltables \
    tqdm \
    visdom \
    natsort \ 
    easydict \
    numpy \
    iopath fvcore pytorch3d ninja 

RUN pip install git+https://github.com/Ambrosiussen/HalpeCOCOAPI.git#subdirectory=PythonAPI
# RUN pip install halpecocotools

FROM install_deps as build_project
WORKDIR /home

RUN mkdir /build


RUN cd /build && git clone https://github.com/MVIG-SJTU/AlphaPose.git
WORKDIR /build/AlphaPose

ENV CUDA_HOME='/usr/local/cuda'
RUN TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" \
    PATH=/usr/local/cuda/bin/:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH \
    python3 setup.py build develop --user

RUN mkdir -p detector/yolox/data && mkdir -p pretrained_models
RUN curl -o detector/yolox/data/yolox_x.pth -L 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth' && \
    curl -o pretrained_models/halpe26_fast_res50_256x192.pth -L 'https://drive.usercontent.google.com/download?id=1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb&export=download&authuser=0&confirm=yes' && \
    curl -o pretrained_models/fast_res50_256x192.pth -L 'https://drive.usercontent.google.com/download?id=1Bb3kPoFFt-M0Y3ceqNO8DTXi1iNDd4gI&export=download&authuser=0&confirm=yes'

