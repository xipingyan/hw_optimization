// Reference:

#include <stdio.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
	printf("== Test USM Memory Extensions Alloc.\n");

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    // 1. 环境初始化 (获取 Intel 平台)
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

    // 2. 加载扩展函数指针
    clDeviceMemAllocINTEL_fn clDeviceMemAllocINTEL =
        (clDeviceMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform, "clDeviceMemAllocINTEL");
    clSharedMemAllocINTEL_fn clSharedMemAllocINTEL =
        (clSharedMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform, "clSharedMemAllocINTEL");
    clMemFreeINTEL_fn clMemFreeINTEL = 
        (clMemFreeINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL");

    if (!clDeviceMemAllocINTEL || !clSharedMemAllocINTEL || !clMemFreeINTEL) {
        printf("该设备不支持 Intel USM 扩展。\n");
        return -1;
    }

    // 3. 分配 Device USM 内存
    size_t size = 1000;
    // device 内存 CPU 不可直接访问，只能通过 clEnqueueMemcpyINTEL 操作
    void* device_ptr = clDeviceMemAllocINTEL(context, device, NULL, size, 0, &err);
    if (err == CL_SUCCESS) {
        printf("成功分配 Device USM 内存，地址: %p\n", device_ptr);
    }

    void* shared_ptr = clSharedMemAllocINTEL(context, device, NULL, size, 0, &err);
    if (err == CL_SUCCESS) {
        printf("成功分配 Shared USM 内存，地址: %p\n", shared_ptr);
    }

    // 4. (可选) 使用内存 - 示例：初始化数据
    int host_data[1024] = {42};
    // 加载 clEnqueueMemcpyINTEL 指针并执行拷贝...
    // 此处省略 Kernel 执行过程，用法与 cl_mem 类似，但参数直接传指针

    // 5. 释放内存, 
    // 观测GPU变化命令： 
    // $ sudo su
    // $ watch -n 0.1 cat /sys/kernel/debug/dri/0/vram0_mm
    clMemFreeINTEL(context, shared_ptr); // shared USM 内存与 host 交互更频繁，释放后驱动会立即更新内存状态
    clMemFreeINTEL(context, device_ptr); // device USM 内存管理由 GPU 驱动异步处理，释放操作可能被延迟，尤其是在 command queue 还未 flush 或 context 未销毁时
    clFinish(queue);

    // 清理资源
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

	printf("== Done.\n");
	return 0;
}