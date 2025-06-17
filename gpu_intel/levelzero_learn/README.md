# Levelzero

Intel OneAPI Levelzero. 
Github: https://github.com/oneapi-src/level-zero
Guide: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html

LevelZero test codes: https://github.com/oneapi-src/level-zero-tests.git

# Build Level-zero

``Windows dependencies``: install spectre v14.1 in VS 2019. Refer [Fix error MSB8040 Guide](https://learn.microsoft.com/en-us/visualstudio/msbuild/errors/msb8040?view=vs-2022)

``Linux``: Add current user to render and video group: ``sudo usermod -a -G render xiping``

    source /opt/intel/oneapi/setvars.sh
    cd CodeSamples\level-zero\
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_INSTALL_PREFIX=install ..
    make -j20 && make install

# Build Samples and Run

    cd CodeSamples
    mkdir build && cd build
    cmake -G"Visual Studio 16" ..

Build all tests:

    cmake --build . --target .\ALL_BUILD --config Release

Build first test:

    cmake --build . --target .\00_HelloLevelZero\00_HelloLevelZero --config Release

Run first case:

    .\00_HelloLevelZero\Release\00_HelloLevelZero.exe