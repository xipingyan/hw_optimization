
cmc_app=../llvm-project/build/install/bin/cmc
kernel_src=./01_HelloCM/src/hello_cm_kernel.cm
out_spv=./build/hello_cm_kernel.spv

$cmc_app -march=SKL $kernel_src -fcmocl -emit-spirv -o $out_spv
cd build
./01_HelloCM/01_HelloCM