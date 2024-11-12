# PTI TOOLS

# onetrace

``Build and Install``

    cd pti-gpu/tools/onetrace
    mkdir build && cd build
    make -j20
    sudo make install
    -- Installing: /usr/local/bin/onetrace
    -- Installing: /usr/local/bin/libonetrace_tool.so

``chrome://tracing``

    onetrace --help | grep chrome 
    onetrace --chrome-call-logging --chrome-device-timeline app 

Refer: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-1/intel-profiling-tools-interfaces-for-gpu.html

