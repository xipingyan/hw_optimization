# HelloOpenCL

Run OpenCL kernel "sample_add", and dump a kernel binary("ocl_kernel.bin").

# How to dump a SPIR-V format kernel

Refer: https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/os_tooling.md

    sudo apt-get install clang
    sudo apt-get install llvm-spirv

    clang -c -target spir -O0 -emit-llvm -o simple_add.bc ../01_HelloOpenCL/src/simple_add.cl
    clang -cl-std=CLC++ -c -target spir -O0 -emit-llvm -o simple_add.bc ../01_HelloOpenCL/src/simple_add.cl

    llvm-spirv simple_add.bc -o sample_add.spv

