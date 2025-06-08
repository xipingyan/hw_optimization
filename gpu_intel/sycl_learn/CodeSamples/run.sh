source /opt/intel/oneapi/setvars.sh

SYCL_MM_PERFORAMNCE=1 ./kernel_01_MatMul/kernel_01_MatMul

SYCL_MM_PERFORAMNCE=1 ONEAPI_DEVICE_SELECTOR=OPENCL:GPU ./kernel_01_MatMul/kernel_01_MatMul