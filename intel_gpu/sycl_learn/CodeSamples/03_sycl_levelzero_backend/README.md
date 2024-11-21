# Readme
This test shows how to co-work between SYCL and other kernels (OpenCL, oneDNN).

#### oneDNN

Build from source codes.

    cd CodeSamples/03_sycl_levelzero_backend
    git clone https://github.com/oneapi-src/oneDNN.git
    cd oneDNN
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DDNNL_GPU_RUNTIME=SYCL -DCMAKE_INSTALL_PREFIX=install ..
    make -j20 && make install

#### Build and run 03

Note, please build oneDNN or install release package.

    cd /hw_optimization/intel_gpu/sycl_learn/CodeSamples/03_sycl_levelzero_backend/
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx ..
    make -j20
    ./03_sycl_levelzero_backend

#### Run this app inside docker(ubuntu22.04)

    <!-- Install driver and add to render group -->
    https://dgpu-docs.intel.com/driver/installation.html#ubuntu

    Start docker command:
    sudo -E docker run \
        -v `pwd`/build/:/workpath \
        -v /opt/intel/oneapi/compiler/2024.2/lib:/opt/intel/oneapi/compiler/2024.2/lib \
        --name test_sycl  \
        --env http_proxy=http://child-prc.intel.com:913 \
        --env https_proxy=http://child-prc.intel.com:913  \
        --device=/dev/dri --security-opt seccomp=unconfined --group-add video \
        -it ubuntu_with_gpu_driver:22.04 \
         /bin/bash

    run inside docker.
    LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2024.2/lib:`pwd` ./03_sycl_levelzero_backend
    # /opt/intel/oneapi/tbb/2021.13/lib/intel64/gcc4.8
    # /opt/intel/oneapi/compiler/2024.2/lib
