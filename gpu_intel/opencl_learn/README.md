# OpenCL

OpenCL: [Guide](https://github.com/KhronosGroup/OpenCL-Guide).

[Optimising OpenCL kernels](https://www.youtube.com/watch?v=OvIrX5XBX8E).

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

# OCL 概念

gws：对应所有的元素
lws：对应一个group的所有元素
group： 例如gws=128， lws=64，group数目为gws/lws=2, 必须为整数倍, 这个数字也可成为group size。
sub_group_size：一个group，也可以分为多有个子的group，kernel中，可以调用''get_sub_group_size();''

```
    const size_t global_id = get_global_id(0);
    const size_t local_id = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t global_ws = get_global_size(0);
    const size_t local_ws = get_local_size(0);

    uint sub_group_size = get_sub_group_size();
    uint sub_group_id = local_id / sub_group_size;
    uint sub_group_lid = local_id % sub_group_size;
```