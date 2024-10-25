# Levelzero

Intel OneAPI Levelzero. 
Github: https://github.com/oneapi-src/level-zero
Guide: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html

# Build Level-zero

    cd CodeSamples\level-zero\
    mkdir build && make install
    cmake -DCMAKE_INSTALL_PREFIX=install ..
    make -j20 && make install

# Build Samples and Run

    cd CodeSamples\00_HelloLevelZero
    mkdir build && cd build
    cmake -G"Visual Studio 16" ..
    cmake --build . --target .\00_HelloLevelZero --config Release

    .\Release\00_HelloLevelZero.exe