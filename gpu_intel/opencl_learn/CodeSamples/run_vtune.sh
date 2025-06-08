cd /mnt/xiping/gpu_profiling/hw_optimization/intel_gpu/opencl_learn/CodeSamples/build

app=./02_Rope_kernel_opencl/02_Rope_kernel_opencl
logs_dir=vtune_log_dir
mkdir -p $logs_dir
PERFORMANCE=1 vtune -collect gpu-hotspots -r $logs_dir -knob gpu-sampling-interval=0.1 -- $app
