# OpenCL mem alloc

Test clDeviceMemAllocINTEL_fn, clSharedMemAllocINTEL_fn, clMemFreeINTEL_fn.

Check GPU memory usage command:

```
$ sudo su
$ watch -n 0.1 cat /sys/kernel/debug/dri/0/vram0_mm
```