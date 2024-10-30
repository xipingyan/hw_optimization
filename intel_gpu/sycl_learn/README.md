# Sycl (DPC++)

SYCL: Download [DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html) and Install.

Github: https://github.com/oneapi-src/level-zero
DPC++ Guide: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2025-0/overview.html

# Build Samples and Run

    cd CodeSamples
    mkdir build && cd build
    cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -G"Visual Studio 16" ..

Build all tests:

    cmake --build . --target .\ALL_BUILD --config Release

Build first test:

    cmake --build . --target .\01_HelloSycl\01_HelloSycl --config Release

Run first case:

    .\01_HelloSycl\Release\01_HelloSycl.exe