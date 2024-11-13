# HelloOpenCL

Run OpenCL kernel "simple_add", and dump a kernel binary("ocl_kernel.bin").

# How to convert to SPIR-V format kernel

Refer: https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/os_tooling.md

    sudo apt-get install clang
    sudo apt-get install llvm-spirv

    clang -c -target spir -O0 -emit-llvm -o simple_add.bc ../01_HelloOpenCL/src/simple_add.cl
    clang -cl-std=CLC++ -c -target spir -O0 -emit-llvm -o simple_add.bc ../01_HelloOpenCL/src/simple_add.cl

    llvm-spirv simple_add.bc -o simple_add.spv

