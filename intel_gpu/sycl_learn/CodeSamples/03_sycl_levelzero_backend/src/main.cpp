// ReadMe:
// Test: SYCL interface + levelzero backend.

// Reference:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md
// https://mlir.llvm.org/doxygen/SyclRuntimeWrappers_8cpp_source.html

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#define PRINT_VAR(var) std::cout << #var << " = " << var << std::endl

namespace
{
    template <typename F>
    auto catchAll(F &&func)
    {
        try
        {
            return func();
        }
        catch (const std::exception &e)
        {
            fprintf(stdout, "An exception was thrown: %s\n", e.what());
            fflush(stdout);
            abort();
        }
        catch (...)
        {
            fprintf(stdout, "An unknown exception was thrown\n");
            fflush(stdout);
            abort();
        }
    }

#define L0_SAFE_CALL(call)                            \
    {                                                 \
        ze_result_t status = (call);                  \
        if (status != ZE_RESULT_SUCCESS)              \
        {                                             \
            fprintf(stdout, "L0 error %d\n", status); \
            fflush(stdout);                           \
            abort();                                  \
        }                                             \
    }

} // namespace

static sycl::device getDefaultDevice()
{
    static sycl::device syclDevice;
    static bool isDeviceInitialised = false;
    if (!isDeviceInitialised)
    {
        auto platformList = sycl::platform::get_platforms();
        for (const auto &platform : platformList)
        {
            auto platformName = platform.get_info<sycl::info::platform::name>();
            bool isLevelZero = platformName.find("Level-Zero") != std::string::npos;
            if (!isLevelZero)
                continue;

            syclDevice = platform.get_devices()[0];
            isDeviceInitialised = true;
            return syclDevice;
        }
        throw std::runtime_error("getDefaultDevice failed");
    }
    else
        return syclDevice;
}

static sycl::context getDefaultContext()
{
    static sycl::context syclContext{getDefaultDevice()};
    return syclContext;
}

static void *allocDeviceMemory(sycl::queue *queue, size_t size, bool isShared)
{
    void *memPtr = nullptr;
    if (isShared)
    {
        memPtr = sycl::aligned_alloc_shared(64, size, getDefaultDevice(),
                                            getDefaultContext());
    }
    else
    {
        memPtr = sycl::aligned_alloc_device(64, size, getDefaultDevice(),
                                            getDefaultContext());
    }
    if (memPtr == nullptr)
    {
        throw std::runtime_error("mem allocation failed!");
    }
    return memPtr;
}
static void deallocDeviceMemory(sycl::queue *queue, void *ptr)
{
    sycl::free(ptr, *queue);
}

static ze_module_handle_t loadModule(const void *data, size_t dataSize)
{
    assert(data);
    ze_module_handle_t zeModule;
    ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             dataSize,
                             (const uint8_t *)data,
                             nullptr,
                             nullptr};
    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        getDefaultDevice());
    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        getDefaultContext());
    L0_SAFE_CALL(zeModuleCreate(zeContext, zeDevice, &desc, &zeModule, nullptr));
    return zeModule;
}

static sycl::kernel *getKernel(ze_module_handle_t zeModule, const char *name)
{
    assert(zeModule);
    assert(name);
    ze_kernel_handle_t zeKernel;
    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;

    L0_SAFE_CALL(zeKernelCreate(zeModule, &desc, &zeKernel));
    sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {zeModule}, getDefaultContext());

    auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {kernelBundle, zeKernel}, getDefaultContext());
    return new sycl::kernel(kernel);
}

static void launchKernel(sycl::queue *queue, sycl::kernel *kernel, size_t gridX,
                         size_t gridY, size_t gridZ, size_t blockX,
                         size_t blockY, size_t blockZ, size_t sharedMemBytes,
                         void **params, size_t paramsCount)
{
    auto syclGlobalRange =
        sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
    auto syclLocalRange = sycl::range<3>(blockZ, blockY, blockX);
    sycl::nd_range<3> syclNdRange(syclGlobalRange, syclLocalRange);

    queue->submit([&](sycl::handler &cgh)
                  {
     for (size_t i = 0; i < paramsCount; i++) {
       cgh.set_arg(static_cast<uint32_t>(i), *(static_cast<void **>(params[i])));
     }
     cgh.parallel_for(syclNdRange, *kernel); });
}

ze_module_handle_t myLoadModule(sycl::context ctxt, sycl::device device, const void *data, size_t dataSize)
{
    assert(data);
    ze_module_handle_t zeModule;
    ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             dataSize,
                             (const uint8_t *)data,
                             nullptr,
                             nullptr};
    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        device);
    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        ctxt);
    L0_SAFE_CALL(zeModuleCreate(zeContext, zeDevice, &desc, &zeModule, nullptr));
    return zeModule;
}

int main()
{
    // Load SPIR-V binary
    std::ifstream spirv_file("path/to/your/spirv_binary.spv", std::ios::binary);
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());

    auto queue = sycl::queue(sycl::gpu_selector_v);
    
    // Create SYCL context and queue using Level Zero backend
    auto context = queue.get_context();
    auto device = queue.get_device();

    auto module = myLoadModule(context, device, nullptr, 20);
    // cl::sycl::context
    // sycl::ext::oneapi::level_zero::
    // cl::sycl::context sc = Q.get_context();

    // // Create SYCL program from SPIR-V binary
    // sycl::program program(context);
    // program.build_with_kernel_type<sycl::kernel>(spirv_binary.data(), spirv_binary.size());

    // // Submit kernel to the queue
    // queue.submit([&](sycl::handler &cgh)
    //              {
    //     auto kernel = program.get_kernel<sycl::kernel>();
    //     cgh.single_task(kernel, [=]() {
    //         // Kernel code here
    //     }); });

    // queue.wait_and_throw();
    // std::cout << "Success!\n";
    return 0;
}