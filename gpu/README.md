# GPU
Learn [video1](https://www.youtube.com/watch?v=6kT7vVHCZIc), [video2](https://www.youtube.com/watch?v=mrDWmnXC5Ck)
Sample [codes](https://cuda-tutorial.github.io/).

#### Video1 Note

1. The free lunch is over
   Three walls: Power wall, Memory wall, ILP wall
2. Today:GPUs without Graphics(Tesla V100, Ampere A100), A parallel powerhouse.
3. CPU vs GPU architecture
   CPU: latency first design, GPU: throughput first design.

#### CodeSamples
Fork from Sample [codes](https://cuda-tutorial.github.io/).

``Dependencies:`` Install cuda driver.

``Build:``

   mkdir build && cd build
   cmake .. && make -j8

``Run:``

   ./01_HelloGPU/01_HelloGPU

# NSight:profiling tool

   NSight is profiling tool, support GPU, ARM, .... <br>
   For Nvidia GPU based on CUPTI interface. <br>
   Install Nsight refer: [nsight-systems/InstallationGuide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#getting-started-guide) <br>
   NVIDIA recommend [Nsight course video](https://resources.nvidia.com/en-us-nsight-developer-tools/nsight-compute)

#### 1. ncu and nuc-ui: (optimize a single grid launch)

      $ ncu --help
      $ ncu --mode=launch ./09_Streams/09_Streams

   `Q&A:` <br>
   ``Error:`` ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. <br>
   ``Solution:`` https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters <br>

   Templorarily solution: <br>

      sudo systemctl isolate multi-user
      sudo modprobe -rf nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
      sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
      sudo systemctl isolate graphical

   See report

      $ ncu-ui <MyReport.ncu-rep>
#### 2. nsys and nsys-ui (profiling CPU/GPU interaction, and GPU concurrency issue.)

   $ nsys profile -o profile_nsys_result --force-overwrite true ./app
   check report
   $ nsys-ui profile_nsys_result.nsys-rep

# Reference
