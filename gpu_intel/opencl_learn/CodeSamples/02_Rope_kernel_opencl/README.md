# HelloOpenCL

Verify OpenCL rope kernel, compare result with SYCL runtime.

# How to run

    cd hw_optimization/intel_gpu/opencl_learn/CodeSamples/
    mkdir build && cd build
    cmake ..
    make -j20
    ./02_Rope_kernel_opencl/02_Rope_kernel_opencl

    == Start compare result with expected.
    Index: 4570 Result 0.0018034!= Expected 0.001804
    == Done.

    If you don't want to print in kernel, please comment printf in kernel file:
    hw_optimization/intel_gpu/sycl_learn/CodeSamples/02_sycl_ocl_interoperate/src/kernel_rope_ref/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl

**My conclusion**: sycl and opencl have same result.