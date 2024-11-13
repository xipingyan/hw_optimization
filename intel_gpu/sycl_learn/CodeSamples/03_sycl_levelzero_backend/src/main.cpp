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

ze_module_handle_t myLoadModule(sycl::context ctxt, sycl::device device, const void *data, size_t dataSize)
{
    std::cout << "  == Start to load module(levelzero)" << std::endl;
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

sycl::kernel *myGetKernel(sycl::context ctxt, ze_module_handle_t zeModule, const char *name)
{
    std::cout << "  == Start to make kernel based on zeModule." << std::endl;
    assert(zeModule);
    assert(name);
    ze_kernel_handle_t zeKernel;
    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;

    L0_SAFE_CALL(zeKernelCreate(zeModule, &desc, &zeKernel));
    sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {zeModule}, ctxt);

    auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {kernelBundle, zeKernel}, ctxt);

    // uint32_t groupSizeX = 32u;
    // uint32_t groupSizeY = 1u;
    // uint32_t groupSizeZ = 1u;
    // zeKernelSuggestGroupSize(zeKernel, 100, 1U, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
    // std::cout << "== suggest group: x=" << groupSizeX << ", y=" << groupSizeY << ", z=" << groupSizeZ << std::endl;
    // zeKernelSetGroupSize(zeKernel, groupSizeX, groupSizeY, groupSizeZ);

    return new sycl::kernel(kernel);
}

void myLaunchKernel(sycl::queue *queue, sycl::kernel *kernel, int element_size,
                    void **params, size_t paramsCount)
{
    std::cout << "  == Start to launch SPIRV kernel(add to sycl::queue)." << std::endl;
    queue->submit([&](sycl::handler &cgh)
                  {
     for (size_t i = 0; i < paramsCount; i++) {
       cgh.set_arg(static_cast<uint32_t>(i), params[i]);
     }

    //  cgh.parallel_for(syclNdRange, *kernel); });
     cgh.parallel_for(sycl::range<1>(element_size), *kernel); });
}

int main()
{
    std::cout << "Start to test call SPIR-V kernel(converted from opencl kernel)." << std::endl;

    // Load SPIR-V binary
    std::string spirv_fn = "../../../opencl_learn/CodeSamples/build/simple_add.spv";
    std::ifstream spirv_file(spirv_fn, std::ios::binary);
    if (!spirv_file.is_open())
    {
        std::cout << "== Fail: Can't open file: " << spirv_fn << std::endl;
        return 0;
    }
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());
    std::cout << "== Readed spirv kernel file: " << spirv_fn << std::endl;

    auto queue = sycl::queue(sycl::gpu_selector_v);

    // Create SYCL context and queue using Level Zero backend
    auto context = queue.get_context();
    auto device = queue.get_device();

    auto module = myLoadModule(context, device, spirv_binary.data(), spirv_binary.size());
    auto kernel = myGetKernel(context, module, "simple_add");

    // input param:
    size_t length = 1000;
    const int32_t xval(1);
    const int32_t yval(2);
    const int32_t bias(3);
    const int32_t expected(xval + yval + bias);

    auto X = sycl::malloc_shared<int32_t>(length, queue);
    auto Y = sycl::malloc_shared<int32_t>(length, queue);
    auto Z = sycl::malloc_shared<int32_t>(length, queue);
    for (size_t i = 0; i < length; i++)
    {
        X[i] = xval;
        Y[i] = yval;
        Z[i] = 0;
    }
    int32_t *params[3] = {X, Y, Z};

    myLaunchKernel(&queue, kernel, length, reinterpret_cast<void **>(params), 3u);

    queue.parallel_for<class sycl_kernel_add_3>(sycl::range<1>(length), [Z](sycl::id<1> i)
                                                { Z[i] = Z[i] + 3; });
    queue.wait();

    bool is_expected = true;
    for (size_t i = 0; i < length; i++)
    {
        if (abs(expected - Z[i]) > 0)
        {
            std::cout << "== Result [" << i << "] diff: " << abs(expected - Z[i]) << ", expect: " << expected << ", result=" << Z[i] << std::endl;
            is_expected = false;
        }
    }

    std::cout << (is_expected ? "Success!\n" : "Fail!\n");
    return 0;
}