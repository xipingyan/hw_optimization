# OpenCL

OpenCL: [Guide](https://github.com/KhronosGroup/OpenCL-Guide).

# Build Samples and Run

#### Windows (git bash)

    cd opencl_learn/CodeSamples

Download OpenCL [SDK](https://github.com/KhronosGroup/OpenCL-SDK/releases) and decompress to OpenCL-SDK path

    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=/c/ov_task/hw_optimization/intel_gpu/opencl_learn/CodeSamples/OpenCL-SDK/ -G"Visual Studio 16" ..

#### Linux

OpenCL is warperred into oneAPI.

    cd opencl_learn/CodeSamples
    source /opt/intel/oneapi/setvars.sh
    mkdir build && cd build
    cmake ..