# ReadMe

Array add or array max, learn to how to use reduction (规约) algorithm for GPU programming.

# How to run

    cd hw_optimization/gpu_intel/opencl_learn/CodeSamples/
    mkdir build && cd build
    cmake ..
    make -j20
    ./04_array_max/04_array_max

# Test result

| kernel name       |  gws       | lws       | time host ms | gpu   | host reduction |
| :---------------- | :----------| :-------  | :------------| :-----| :--------------|
| get_array_max_1   | [7680,1,1] | [32,1,1]  | 234          | 231   | 2              |
| get_array_max_2   | [7680,1,1] | [32,1,1]  | 222          | 218   | 3              |
