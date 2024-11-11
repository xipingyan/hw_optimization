# Sycl (DPC++)

SYCL: Download [DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html) and Install. <br>

Github: https://github.com/oneapi-src/level-zero <br>
DPC++ Guide: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2025-0/overview.html <br>

``Keynotes:`` <br>

    1: SYCL is definitely modern C++(cpp11, cpp17)
    2: OpenMP 5.0 provides same feature as SYCL and DPC++, but it is based on C/Fortan/premodern CPP.
    3: SYCL is derived from OpenCL technology, and the run models are similar. 


``Programming guide``: <br>

    0: https://link.springer.com/book/10.1007/978-1-4842-5574-2 [pdf e-book]
    1: https://www.intel.com/content/www/us/en/developer/articles/training/programming-data-parallel-c.html [Done]
    2: https://developer.codeplay.com/products/computecpp/ce/2.11.0/guides/sycl-guide/hello-sycl 
    3: (in-order or out-of-order) https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/executing-multiple-kernels-on-the-device-at-the.html
    4: Architecture: https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html
    4.1 https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/intel-xe-gpu-architecture.html
    
    5: https://github.com/intel/llvm-test-suite/blob/intel/SYCL [All Sycl grammar test?]

# Build Samples and Run

#### Windows(Not work)

    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -G"Visual Studio 16" ..

#### Linux (Dump SPIR-V)

``Linux``: Add current user to render and video group: ``sudo usermod -a -G render xiping``

    source /opt/intel/oneapi/setvars.sh
    cd CodeSamples
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=icpx ..      # Sycl need to use icpx (intel compiler)
    SYCL_DUMP_IMAGES=1  ./01_HelloSycl/01_HelloSycl

#### Parse spv file(SPIR-V format)

``1`` SPIRV-ToolsS

    cd hw_optimization/intel_gpu/
    git clone https://github.com/KhronosGroup/SPIRV-Tools
    python3 utils/git-sync-deps
    mkdir build && cd build
    cmake .. && make -j20
    ../../../SPIRV-Tools/build/tools/spirv-dis ./sycl_spir64.spv --comment > dump.txt

``2`` Visulizer online: https://www.khronos.org/spir/visualizer/
