# OpenCL mem alloc

Test clDeviceMemAllocINTEL_fn, clSharedMemAllocINTEL_fn, clMemFreeINTEL_fn.

Check GPU memory usage command:

```
$ sudo su
$ watch -n 0.1 cat /sys/kernel/debug/dri/0/vram0_mm
```

观测命令：xpu-manager
```
watch -n 1 "sudo xpu-smi stats -d 0"
```

``Note``: Test found: for device memory, only alloc > 256M, it can be observed via xpu-manger.