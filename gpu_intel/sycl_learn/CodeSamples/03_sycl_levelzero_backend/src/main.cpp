// ReadMe:
// Test: SYCL interface + levelzero backend.

// Reference:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md
// https://mlir.llvm.org/doxygen/SyclRuntimeWrappers_8cpp_source.html

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

// oneDNN headers
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include "my_common.hpp"

#define ENABLE_LEVELZERO 0
#if ENABLE_LEVELZERO
#include <level_zero/ze_api.h>

namespace
{
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

sycl::event myLaunchKernel(sycl::queue *queue, sycl::kernel *kernel, int element_size,
                           void **params, size_t paramsCount)
{
    std::cout << "  == Start to launch SPIRV kernel(add to sycl::queue)." << std::endl;
    return queue->submit([&](sycl::handler &cgh)
                         {
     for (size_t i = 0; i < paramsCount; i++) {
       cgh.set_arg(static_cast<uint32_t>(i), params[i]);
     }

    //  cgh.parallel_for(syclNdRange, *kernel); });
     cgh.parallel_for(sycl::range<1>(element_size), *kernel); });
}

// Launch SPIR-V format kernel (Converted from OpenCL)
// Converted OpenCL to SPIR-V:
// Refer: https://github.com/xipingyan/hw_optimization/blob/main/intel_gpu/opencl_learn/CodeSamples/01_HelloOpenCL/README.md
sycl::event launchSPVKernelFromOpenCLOffline(sycl::queue &queue, size_t length, int32_t *X, int32_t *Y, int32_t *Z)
{
    std::cout << "== Start to launch SPIR-V kernel(converted from opencl kernel)." << std::endl;

    // Load SPIR-V binary
    std::string spirv_fn = "../../../opencl_learn/CodeSamples/build/simple_add.spv";
    std::ifstream spirv_file(spirv_fn, std::ios::binary);
    if (!spirv_file.is_open())
    {
        std::cout << "== Fail: Can't open file: " << spirv_fn << std::endl;
        exit(0);
    }
    std::vector<char> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());
    std::cout << "  == Readed spirv kernel file: " << spirv_fn << std::endl;

    // Create SYCL context and queue using Level Zero backend
    auto context = queue.get_context();
    auto device = queue.get_device();

    auto module = myLoadModule(context, device, spirv_binary.data(), spirv_binary.size());
    auto kernel = myGetKernel(context, module, "simple_add");

    std::cout << "  == launch kernel" << std::endl;
    int32_t *params[3] = {X, Y, Z};
    return myLaunchKernel(&queue, kernel, length, reinterpret_cast<void **>(params), 3u);
}

// Way 2: Launch SPIR-V format kernel (Converted from OpenCL)
// Refer : https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler_spirv.asciidoc
// 2 is more friendly, but it alsp trigger other kernel crash.???
sycl::event launchSPVKernelFromOpenCLOffline_2(sycl::queue &q, size_t length, int32_t *X, int32_t *Y, int32_t *Z)
{
    std::cout << "== Start to launch SPIR-V kernel(converted from opencl kernel)." << std::endl;

    // Load SPIR-V binary
    std::string spirv_fn = "../../../opencl_learn/CodeSamples/build/simple_add.spv";
    std::cout << "  == Start to SPIR-V kernel file: " << spirv_fn << std::endl;
    // Read the SPIR-V module from disk.
    std::ifstream spv_stream(spirv_fn, std::ios::binary);
    spv_stream.seekg(0, std::ios::end);
    size_t sz = spv_stream.tellg();
    spv_stream.seekg(0);
    std::vector<std::byte> spv(sz);
    spv_stream.read((char *)spv.data(), sz);

    // Create a kernel bundle from the binary SPIR-V.
    std::cout << "  == Start to kernel_bundle spv" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(
            q.get_context(),
            syclex::source_language::spirv,
            spv);

    // Build the SPIR-V module for our device.
    std::cout << "  == Start to build kb_src" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclex::build(kb_src);

    // Get a "kernel" object representing the kernel from the SPIR-V module.
    std::cout << "  == Start to get sycl::kernel" << std::endl;
    sycl::kernel k = kb_exe.ext_oneapi_get_kernel("simple_add");

    // constexpr int N = 4;
    constexpr int WGSIZE = 1;
    // int32_t input[N] = {0, 1, 2, 3};
    // int32_t output[N] = {};

    sycl::buffer buf_x(X, sycl::range{length});
    sycl::buffer buf_y(Y, sycl::range{length});
    sycl::buffer buf_z(Z, sycl::range{length});

    std::cout << "  == Start to submit kernel" << std::endl;
    return q.submit([&](sycl::handler &cgh)
                    {
                        sycl::accessor in_x{buf_x, cgh, sycl::read_only};
                        sycl::accessor in_y{buf_y, cgh, sycl::read_only};
                        sycl::accessor out_z{buf_z, cgh, sycl::read_write};

                        // Set the values for the kernel arguments.
                        cgh.set_args(in_x, in_y, out_z);

                        // Invoke the kernel over an nd-range.
                        sycl::nd_range ndr{{length}, {WGSIZE}};
                        cgh.parallel_for(ndr, k); });
}
#endif

class sycl_args
{
public:
    sycl_args() = delete;
    sycl_args(sycl::buffer<uint8_t, 1, sycl::image_allocator, void> buf, bool isOutput) : _isBuf(true), _buf(buf), _val(0), _isOutput(isOutput)
    {
    }
    sycl_args(int val) : _isBuf(false), _buf(0, 1), _val(val)
    {
    }
    bool _isBuf;
    sycl::buffer<uint8_t, 1, sycl::image_allocator, void> _buf;
    int _val = 0; // if isBuf == false;
    bool _isOutput = false;
    friend std::ostream &operator<<(std::ostream &os, const sycl_args &bf);
};

std::ostream &operator<<(std::ostream &os, const sycl_args &bf)
{
    os << "sycl_args(_isBuf = " << bf._isBuf << ", _val = " << bf._val << ", _isOutput = " << bf._isOutput << ")";
    return os;
};

void my_set_args(sycl::handler &cgh, size_t idx, sycl_args buf) {
    if (buf._isOutput)
    {
        // Last one is output.
        sycl::accessor acc_param{buf._buf, cgh, sycl::read_write};
        cgh.set_arg(idx, acc_param);
    }
    else
    {
        if (buf._isBuf)
        {
            sycl::accessor acc_param{buf._buf, cgh, sycl::read_only};
            cgh.set_arg(idx, acc_param);
        }
        else
        {
            cgh.set_arg(idx, buf._val);
        }
    }
}

// Launch OpenCL, online compile to Sycl interface.
// Refer: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler_opencl.asciidoc
sycl::event launchOpenCLKernelOnline(sycl::queue &q, size_t length, int32_t *X, int32_t *Z, int32_t offset, sycl::event &dep_event)
{
    std::cout << "== Start to test launch OpenCL kernel and compile online." << std::endl;
    // Kernel defined as an OpenCL C string.  This could be dynamically
    // generated instead of a literal.
    std::string source = R"""(
        __kernel void my_kernel(__global int *in, __global int *out, int offset_val) {
            size_t i = get_global_id(0);
            out[i] = out[i]*2 + in[i] + offset_val;
            // printf("  == offset_val = %d, i = %d\n", offset_val, i);
        }
    )""";

    std::cout << "  == Start to kernel_bundle opencl source" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(
            q.get_context(),
            syclex::source_language::opencl,
            source);

    // Compile and link the kernel from the source definition.
    std::cout << "  == Start to kernel_bundle kb_src" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclex::build(kb_src);

    // Get a "kernel" object representing the kernel defined in the
    // source string.
    std::cout << "  == Start to get sycl::kernel" << std::endl;
    sycl::kernel k = kb_exe.ext_oneapi_get_kernel("my_kernel");

    // constexpr int N = length;
    constexpr int WGSIZE = 1;

#define UNIFY_DATA_TYPE 1
#if UNIFY_DATA_TYPE
    sycl::buffer inputbuf((uint8_t*)X, sycl::range{length*sizeof(int32_t)});
    sycl::buffer outputbuf((uint8_t*)Z, sycl::range{length*sizeof(int32_t)});

    std::vector<sycl_args> inputs_buf;
    inputs_buf.push_back(sycl_args(inputbuf, false));
    inputs_buf.push_back(sycl_args(outputbuf, true));
    inputs_buf.push_back(sycl_args(offset));
#else
    sycl::buffer inputbuf(X, sycl::range{length});
    sycl::buffer outputbuf(Z, sycl::range{length});
#endif
    std::cout << "  == Start to submit" << std::endl;

    // for (int i = 0; i < inputs_buf.size(); i++) {
    //     std::cout << "inputs[" << i << "] = " << inputs_buf[i] << std::endl;
    // }
    return q.submit([&](sycl::handler &cgh)
                    {
                        cgh.depends_on(dep_event);
#if UNIFY_DATA_TYPE
                        for (int i = 0; i < inputs_buf.size(); i++)
                        {
                            my_set_args(cgh, i, inputs_buf[i]);
                        }
#else
                        sycl::accessor in{inputbuf, cgh, sycl::read_only};
                        sycl::accessor out{outputbuf, cgh, sycl::read_write};
                        cgh.set_args(in, out); // All arguments
                        cgh.set_arg(2, offset); // scalar param
#endif
                        // Invoke the kernel over an nd-range.
                        sycl::nd_range ndr{{length}, {WGSIZE}};
                        cgh.parallel_for(ndr, k);
                    });
}

void launchOneDNNKernel_reference(std::vector<int32_t> &expected)
{
    for (size_t i = 0; i < expected.size(); i++)
    {
        expected[i] = expected[i] < 0 ? 0 : expected[i];
    }
}

// Refer:
// https://github.com/oneapi-src/oneDNN/blob/main/examples/sycl_interop_usm.cpp
// https://oneapi-src.github.io/oneDNN/v2/dev_guide_dpcpp_interoperability.html
sycl::event launchOneDNNKernel(sycl::queue &q, size_t length, int32_t *Z, sycl::event &dep_event)
{
    auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());

    auto strm = dnnl::sycl_interop::make_stream(eng, q);

    auto usm_buffer = (float *)malloc_shared(length * sizeof(float),
                                             dnnl::sycl_interop::get_device(eng), dnnl::sycl_interop::get_context(eng));

    dnnl::memory::dims tz_dims = {1, 1, 1, static_cast<dnnl_dim_t>(length)};
    // dnnl::memory::desc mem_d(tz_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
    dnnl::memory::desc mem_d(tz_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nchw);

    dnnl::memory mem = dnnl::sycl_interop::make_memory(mem_d, eng, dnnl::sycl_interop::memory_kind::usm, Z);

    float alpha = 0.0f;
    auto relu_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward,
                                                         dnnl::algorithm::eltwise_relu, mem_d, mem_d, alpha);
    auto relu = dnnl::eltwise_forward(relu_pd);

    return dnnl::sycl_interop::execute(relu, strm, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}}, {dep_event});
    // relu_e.wait();
}

int main()
{
    std::cout << "Start to test call SPIR-V kernel(converted from opencl kernel)." << std::endl;

    auto queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::cout << "== Using "
              << queue.get_device().get_info<sycl::info::device::name>()
              << ", Backend: " << queue.get_backend()
              << std::endl;
    auto order_properity = queue.get_property<sycl::property::queue::in_order>();
    std::cout << "  == order_properity = " << order_properity.getKind() << std::endl;

    // input param:
    size_t length = 10;
#define USM_BUF_QUEUE 0
#if USM_BUF_QUEUE
    auto X = sycl::malloc_shared<int32_t>(length, queue);
    auto Y = sycl::malloc_shared<int32_t>(length, queue);
    auto Z = sycl::malloc_shared<int32_t>(length, queue);
#else
    sycl::context ctx = queue.get_context();
    sycl::device dev = queue.get_device();
    auto X = sycl::malloc_shared<int32_t>(length, dev, ctx);
    auto Y = sycl::malloc_shared<int32_t>(length, dev, ctx);
    int32_t *Z = (int32_t*)(sycl::malloc_shared(length * sizeof(int32_t), dev, ctx));
#endif

    auto expected = std::vector<int32_t>(length);
    for (size_t i = 0; i < length; i++)
    {
        X[i] = (i % 2) ? (i % 100) : (-i % 100);
        Y[i] = X[i];
        Z[i] = 0;
    }

#if ENABLE_LEVELZERO
    // 1: OpenCL offline kernel: Z = X + Y;
    for (size_t i = 0; i < length; i++)
    {
        expected[i] = X[i] + Y[i];
    }
    auto event1 = launchSPVKernelFromOpenCLOffline(queue, length, X, Y, Z);
    #else
    auto event1 = sycl::event();
#endif

    // 2: OpenCL online kernel: Z = 2 * Z + X;
    int32_t offset = 2;
    for (size_t i = 0; i < length; i++)
    {
        expected[i] = 2 * expected[i] + X[i] + offset;
    }
    auto event2 = launchOpenCLKernelOnline(queue, length, X, Z, offset, event1);

    // 3: Sycl kernel: Z = Z + 3
    for (size_t i = 0; i < length; i++)
    {
        expected[i] = expected[i] + 3;
    }
    std::cout << "== Start to launch native sycl kernel." << std::endl;
#if 1
    event2.wait();
    auto event3 = queue.parallel_for<class sycl_kernel_add_3>(sycl::range<1>(length), [Z](sycl::id<1> i)
                                                              { Z[i] = Z[i] + 3; });
#else // Add depends_on.
    auto event3 = queue.submit([&](sycl::handler &cgh)
                               {
                                    cgh.depends_on(event2);
                                    // Invoke the kernel over an nd-range.
                                    sycl::nd_range ndr{{length}, {1}};
                                    cgh.parallel_for(sycl::range<1>(length), [Z](sycl::id<1> i){
                                        Z[i] = Z[i] + 3;
                                    }); });
#endif

    // 4: oneDNN kernel
    std::cout << "== Launch oneDNN kernel..." << std::endl;
    launchOneDNNKernel_reference(expected);
    auto envet4 = launchOneDNNKernel(queue, length, Z, event3);

    // 5: SYCL kernel
    std::cout << "== Launch sycl_kernel_add_y." << std::endl;
    for (size_t i = 0; i < length; i++)
    {
        expected[i] = expected[i] + Y[i];
    }
    queue.parallel_for<class sycl_kernel_add_y>(sycl::range<1>(length), [Z, Y](sycl::id<1> i)
                                                { Z[i] = Z[i] + Y[i]; });

    queue.wait();

    std::cout << "== Start to compare result." << std::endl;
    bool is_expected = true;
    for (size_t i = 0; i < length; i++)
    {
        if (abs(expected[i] - Z[i]) > 0)
        {
            std::cout << "== Result [" << i << "] diff: " << abs(expected[i] - Z[i]) << ", expect: " << expected[i] << ", result=" << Z[i] << std::endl;
            is_expected = false;
        }
    }

    sycl::device get_dev = sycl::get_pointer_device(X, ctx);
    auto y_type = sycl::get_pointer_type(Y, ctx);
    std::cout << "  == Check original device and queryed device from buffer pointer is same: " << (get_dev == dev) << std::endl;
    std::cout << "  == Query Y usm alloc type: " << usm_alloc_2_str(y_type) << std::endl;

#if USM_BUF_QUEUE
    sycl::free(X, queue);
    sycl::free(Y, queue);
    sycl::free(Z, queue);
#else
    sycl::free(X, ctx);
    sycl::free(Y, ctx);
    sycl::free(Z, ctx);
#endif

    std::cout << (is_expected ? "Success!\n" : "Fail!\n");
    return 0;
}