# https://hub.docker.com/r/cwaffles/openpose
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

#replace cmake as old version has CUDA variable bugs
RUN echo "Installing CMake..." && wget https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6.tar.gz && \
tar xf cmake-3.19.6.tar.gz -C /opt && \
rm cmake-3.19.6.tar.gz
WORKDIR /cmake-3.19.6
RUN ./configure && make && make install
# ENV PATH="/opt/cmake-3.19.6-Linux-x86_64/bin:${PATH}"

#get deps
RUN echo "Installing dependencies..." && apt-get --assume-yes update && \
DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install -y --no-install-recommends \
python3-dev python3-pip python3-setuptools build-essential git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev \
liblmdb-dev libgflags-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev

#for python api
RUN pip3 install --upgrade pip protobuf numpy opencv-python
# RUN pip3 install numpy opencv-python  

#get openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git . 

#build openpose
WORKDIR /openpose
RUN mkdir build && cd build && cmake -DBUILD_PYTHON=ON .. && make -j `nproc` && mkdir output