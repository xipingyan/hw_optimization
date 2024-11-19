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