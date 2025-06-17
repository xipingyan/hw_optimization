# CM (C for Metal)

Intel's C for Metal (CM) is a programming language designed to allow developers to achieve close-to-assembly performance on Intel Processor Graphics (including Arc and Xe GPUs). It's a low-level, explicit SIMD (Single Instruction, Multiple Data) programming model that offers fine-grained control over the GPU hardware. <br>


``Key Features:`` <br>
``Explicit SIMD:`` CM exposes the SIMD architecture of Intel GPUs, allowing you to explicitly manage vector and matrix operations.
``Close-to-Metal:`` It provides intrinsics and language constructs that map closely to the GPU's native instruction set (Gen ISA).
``Compatibility:`` CM kernels are compatible with Intel GPU OpenCL runtime and oneAPI Level Zero, allowing them to be launched like OpenCL kernels.
``CPU Emulation:`` There's a "cm-emulation" tool that lets you develop and debug CM kernels on a host CPU, which can be very useful for initial development and testing.

``Data types``<br>
``1:`` explicit SIMD programming model; <br>
``2:`` Vector data types—vector, matrix, and surface; <br>
``3:`` vector operations on these data types; <br>
``4:`` vector conditions if / else, independently performed for each element of the vector; <br>
``5:`` built-in features for accessing Intel GPU hardware fixed functionality; <br>


``Programming guide``: <br>

    0: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-c-for-metal-the-precious-metal-for-computing-on-intel-graphics-cards.html

    1: https://github.com/intel/cm-compiler/blob/cmc_monorepo_110/clang/Readme.md
    2: https://github.com/intel/cm-compiler



# HW and SW dependencies

    groups
    video users render

#### build cmc from source code.
    
    cd ./cm_learn/

    ROOT=`pwd`
    git clone https://github.com/intel/cm-compiler.git -b cmc_monorepo_110 llvm-project
    git clone https://github.com/intel/vc-intrinsics.git llvm-project/llvm/projects/vc-intrinsics
    git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git -b llvm_release_110 llvm-project/llvm/projects/SPIRV-LLVM-Translator

    cd llvm-project && mkdir build && cd build
    BUILD=`pwd`
    INSTALL=`pwd`/install

    cmake -DLLVM_ENABLE_Z3_SOLVER=OFF -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="" $ROOT/llvm-project/llvm
    make -j32 
    make install # very slow.

    <!-- cmc compile cm kernel example refer -->
    https://github.com/intel/cm-compiler/tree/cmc_monorepo_110/clang#running-the-compiler

#### levelzero(just for runtime)

    refer: ../levelzero_learn/README.md [Build Level-zero]
    code workpath: cm_learn/CodeSamples/level-zero

# Build Samples and Run

#### Linux

``Linux``: Add current user to render and video group: ``sudo usermod -a -G render xiping``

    source /opt/intel/oneapi/setvars.sh
    cd CodeSamples
    mkdir build && cd build
    cmake ..
    make -j32
    
    <!-- convert cm kernel to spv -->
    ../../llvm-project/build/install/bin/cmc -march=SKL ../01_HelloCM/src/hello_cm_kernel.cm -fcmocl -emit-spirv -o hello_cm_kernel.spv

    <!-- run cm kernel -->
    ./01_HelloCM/01_HelloCM

# Profiling kernel tool

Refer: sycl_learn/README.md [Tools]