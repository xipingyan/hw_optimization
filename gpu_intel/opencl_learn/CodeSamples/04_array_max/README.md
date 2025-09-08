# ReadMe

Array add or array max, learn to how to use reduction (规约) algorithm for GPU programming.

# How to run

    cd hw_optimization/gpu_intel/opencl_learn/CodeSamples/
    mkdir build && cd build
    cmake ..
    make -j20
    ./04_array_max/04_array_max

# Test result

It's hard to get stable time on host, just statistic kernel time via onetrace. <br>

Time unit: micro second.    <br>

| kernel name       |  gws       | lws       | onetrace kernel time |
| :---------------- | :----------| :-------  | :--------------------|
| get_array_max_1   | [7680,1,1] | [32,1,1]  | 86                   |
| get_array_max_2   | [7680,1,1] | [32,1,1]  | 24                   |
| get_array_max_3   | [7680,1,1] | [64,1,1]  | 20                   |
