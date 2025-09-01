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

``__local``: local内存，使用时，setargs只要指定内存大小即可，不可使用host内存初始化。当你将一个 cl::Buffer 对象传递给 __local 参数时，OpenCL 驱动程序会将其解释为一个普通的 __global 指针，而不是 __local 内存. <br>

```
kernel.setArg(2, sizeof(int) * lws, nullptr);
```

``Reduction``: 归约算法，需要同步，OpenCL支持一个group内同步，所以归约算法必须在一个group内完成。参考kernel: ``get_array_max_single_group``