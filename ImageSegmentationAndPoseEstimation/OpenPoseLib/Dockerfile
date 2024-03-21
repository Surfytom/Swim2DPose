FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

#get deps
RUN echo "Installing dependencies..." \
&& apt-get --assume-yes update && \
DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install -y --no-install-recommends \
python3-dev python3-pip python3-setuptools build-essential git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev \
liblmdb-dev libgflags-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev libssl-dev libgtk2.0-dev pkg-config unzip

RUN echo "Installing CMake..." && \
wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

#for python api
RUN pip3 install --upgrade pip protobuf numpy opencv-python gdown

#get openpose
WORKDIR /openpose
RUN rm -rf openpose.git
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git . 

# download and copy models
WORKDIR /
RUN gdown https://drive.google.com/uc?id=1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh && unzip models.zip
RUN mv models/hand/pose_iter_102000.caffemodel openpose/models/hand
RUN mv models/face/pose_iter_116000.caffemodel openpose/models/face
RUN mv models/pose/coco/pose_iter_440000.caffemodel openpose/models/pose/coco
RUN mv models/pose/mpi/pose_iter_160000.caffemodel openpose/models/pose/mpi
RUN mv models/pose/body_25/pose_iter_584000.caffemodel openpose/models/pose/body_25

#build openpose
WORKDIR /openpose
RUN echo "Installing CMake..."
RUN mkdir build 

WORKDIR /openpose/build
RUN cmake .. && make -j `nproc`
WORKDIR /openpose
RUN mkdir output