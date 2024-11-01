# Sycl (DPC++)

SYCL: Download [DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html) and Install.

Github: https://github.com/oneapi-src/level-zero
DPC++ Guide: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2025-0/overview.html

# Build Samples and Run

#### Windows(Not work)

    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -G"Visual Studio 16" ..

#### Linux

``Linux``: Add current user to render and video group: ``sudo usermod -a -G render xiping``

    source /opt/intel/oneapi/setvars.sh
    cd CodeSamples
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=icpx ..      # Sycl need to use icpx (intel compiler)
    SYCL_DUMP_IMAGES=1  ./01_HelloSycl/01_HelloSycl

#### Parse spv

``1`` SPIRV-ToolsS

    cd hw_optimization/intel_gpu/
    git clone https://github.com/KhronosGroup/SPIRV-Tools
    python3 utils/git-sync-deps
    mkdir build && cd build
    cmake .. && make -j20
    ../../../SPIRV-Tools/build/tools/spirv-dis ./sycl_spir64.spv --comment > dump.txt

``2`` Visulizer online: https://www.khronos.org/spir/visualizer/
