# 03_DPP_algo

Try to write dpp algorithm based on OpenCL kernel.

# How to run

    cd hw_optimization/intel_gpu/opencl_learn/CodeSamples/
    mkdir build && cd build
    cmake ..
    make -j20
    ./03_DPP_algo/03_DPP_algo

# Performance

B=Batch, M=Token number, selected rate=50%. <br>

| GPU                         |Driver         | B  | M    | Time ms | CPU         | CPU time |
|:----------------------------|:--------------|:--:|:----:|:--------|:----------- | ------|
|Intel(R) Iris(R) Xe Graphics | 32.0.101.6979 | 1  | 1792 | 94 (66) | i7-1270P    | 38    |
|Intel(R) Iris(R) Xe Graphics | 32.0.101.6979 | 2  | 1792 | 114(100)| i7-1270P    | 74    |
|Intel(R) Iris(R) Xe Graphics | 32.0.101.6979 | 1  | 3584 | 570(570)| i7-1270P    | 117   |
|B580                         | 25.18.33578   | 1  | 1792 | 29      | i9-14900K   |   139 |
|B580                         | 25.18.33578   | 2  | 1792 | 30      | i9-14900K   |   259 |
|B580                         | 25.18.33578   | 1  | 3584 | 113     | i9-14900K   |   431 |
|A770                         | 25.18.33578   | 1  | 1792 | 29      | XEON(R) 8592+ | 460  |
|A770                         | 25.18.33578   | 2  | 1792 | 32      | XEON(R) 8592+ | 918  |
|A770                         | 25.18.33578   | 1  | 3584 | 131     | XEON(R) 8592+ | 1329 |

A770/B580: <br>

    argmax_gws = [1, 1024, 1]
    argmax_lws = [1, 1024, 1]
    gws = [1, 1792, 1]
    lws = [1, 16, 1]

Intel(R) Iris(R) Xe Graphics <br>

    argmax_gws = [2, 512, 1]
    argmax_lws = [1, 512, 1]
    gws = [2, 1792, 1]
    lws = [1, 16, 1]

For B580, Driver 24.52.32224 and 25.18.33578 have similar performance. <br>

#### Profiling GPU kernel found：

``1:`` Kernel time is related to CPU ``enqueueNDRangeKernel``, some drivers have better performance(driver:25.18.33578) <br>
``2:`` Pure kernel time is about 10ms. (from onetrace on the B580) <br>
