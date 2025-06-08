source /opt/intel/oneapi/setvars.sh
rm -rf onetrace.*
# PERFORMANCE=1 onetrace --chrome-call-logging --chrome-device-timeline ./02_sycl_ocl_interoperate
PERFORMANCE=1 ./02_sycl_ocl_interoperate
