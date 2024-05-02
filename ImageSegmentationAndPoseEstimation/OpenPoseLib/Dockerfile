FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 as install_1

#get deps   
RUN echo "Installing dependencies..." \
    && apt-get --assume-yes update && \
    DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install --no-install-recommends libboost-all-dev \
    && apt-get --assume-yes install \
    cudnn9-cuda-11 \
    libopencv-dev \
    libatlas-base-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    git \
    g++ \
    make \
    opencl-headers \
    ocl-icd-opencl-dev \
    libviennacl-dev \
    libssl-dev \
    libgtk2.0-dev \
    pkg-config unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM install_1 as install_2
RUN echo "Installing CMake..." && \
    curl -o cmake-3.19.6-Linux-x86_64.tar.gz -L 'https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-x86_64.tar.gz' && \
    tar xzf cmake-3.19.6-Linux-x86_64.tar.gz -C /opt && \ 
    rm cmake-3.19.6-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.19.6-Linux-x86_64/bin:${PATH}"


#for python api
RUN pip3 install --upgrade pip protobuf numpy opencv-python

FROM install_2 as clone_openpose
#get openpose
WORKDIR /openpose
RUN rm -f openpose.git   

FROM clone_openpose as download_model        
# download and copy models
WORKDIR /
RUN git clone --depth 1 -b v1.7.0 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
RUN curl -o pretrained_models/fast_res50_256x192.pth -L 'https://drive.usercontent.google.com/download?id=1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh&export=download&authuser=0&confirm=yes' \ && unzip models.zip 
RUN mv models/hand/pose_iter_102000.caffemodel openpose/models/hand \
    && mv models/face/pose_iter_116000.caffemodel openpose/models/face \
    && mv models/pose/coco/pose_iter_440000.caffemodel openpose/models/pose/coco \
    && mv models/pose/mpi/pose_iter_160000.caffemodel openpose/models/pose/mpi \
    && mv models/pose/body_25/pose_iter_584000.caffemodel openpose/models/pose/body_25 \
    && rm -rf models.zip && rm -rf models

FROM download_model as build_openpose
#build openpose
WORKDIR /openpose
RUN echo "Building OpenPose..." && mkdir build 

WORKDIR /openpose/build
RUN cmake .. && make -j `nproc`
WORKDIR /openpose
RUN mkdir output

