sudo -E docker run \
        -v `pwd`/build/:/workpath \
        -v /opt/intel/oneapi/tbb/2021.13/lib/intel64/gcc4.8:/opt/intel/oneapi/tbb/2021.13/lib/intel64/gcc4.8 \
        -v /opt/intel/oneapi/compiler/2024.2/lib:/opt/intel/oneapi/compiler/2024.2/lib \
        --name test_sycl  \
        --env http_proxy=http://child-prc.intel.com:913 \
        --env https_proxy=http://child-prc.intel.com:913  \
	--device=/dev/dri --security-opt seccomp=unconfined --group-add video \
        -it ubuntu_with_gpu_driver:22.04 \
         /bin/bash
