// Reference:
// https://www.intel.com/content/www/us/en/developer/articles/technical/sycl-interoperability-study-opencl-kernel-in-dpc.html

// How to use GPU?
// $ export  ONEAPI_DEVICE_SELECTOR=OPENCL:GPU 
// $ ./02_sycl_ocl_interoperate/02_sycl_ocl_interoperate

#include <iostream>
#include <cmath>
#include <sycl/sycl.hpp>

// OpenCL Kernel
auto add_ocl_kernel_add_1(cl_context ocl_ctx, cl_device_id ocl_dev, sycl::context ctx, sycl::buffer<int, 1>& buffer, size_t size, sycl::queue& q) {
    cl_int err = CL_SUCCESS;

    const char *kernelSource =
        R"CLC(kernel void add(global int* data) {
            int index = get_global_id(0);
            data[index] = data[index] + 1;
        }
        )CLC";

    cl_program ocl_program = clCreateProgramWithSource(ocl_ctx, 1, &kernelSource, nullptr, &err);
    clBuildProgram(ocl_program, 1, &ocl_dev, nullptr, nullptr, nullptr);

    cl_kernel ocl_kernel = clCreateKernel(ocl_program, "add", nullptr);
    sycl::kernel add_kernel = sycl::make_kernel<sycl::backend::opencl>(ocl_kernel, ctx);

    return q.submit([&](sycl::handler &h)
                    {
                auto data_acc =buffer.get_access<sycl::access_mode::read_write, sycl::target::device>(h);
                h.set_args(data_acc);
                h.parallel_for(size, add_kernel); });
}

// Sycl Kernel
auto sycl_kerenel_mul2(sycl::buffer<int, 1> &buffer, size_t sz, sycl::queue &queue)
{
    return queue.submit([&](sycl::handler &cgh)
                        {
        auto outAcc = buffer.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(sz), [=](sycl::id<1> index)
                                       { outAcc[index] = outAcc[index] * 2; }); });
}

// Sycl Kernel
auto sycl_kerenel_add_2(sycl::buffer<int, 1> &buffer, size_t sz, sycl::queue &queue)
{
    return queue.submit([&](sycl::handler &cgh)
                        {
        auto outAcc = buffer.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(sz), [=](sycl::id<1> index)
                                       { outAcc[index] = outAcc[index] + 2; 
                                         float v = 0;
                                         for(uint i=0;i<1000000;i++){
                                             v = sin(i) + v;
                                         }}); });
}

int main()
{
    constexpr size_t size = 16;
    std::array<int, size> data;

    for (int i = 0; i < size; i++)
    {
        data[i] = i;
    }
    char env[] = "ONEAPI_DEVICE_SELECTOR=OPENCL:GPU";
    putenv(env);

    // Need to set ENV: ONEAPI_DEVICE_SELECTOR=OPENCL:GPU
    // sycl::device dev(sycl::default_selector{});
    sycl::device dev(sycl::gpu_selector_v);
    sycl::context ctx = sycl::context(dev);

    auto ocl_dev = sycl::get_native<sycl::backend::opencl, sycl::device>(dev);
    auto ocl_ctx = sycl::get_native<sycl::backend::opencl, sycl::context>(ctx);

    cl_int err = CL_SUCCESS;
    cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(ocl_ctx, ocl_dev, 0, &err);
    sycl::queue q = sycl::make_queue<sycl::backend::opencl>(ocl_queue, ctx);

    std::cout << "Using device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << ", Backend: " << q.get_backend()
              << std::endl;

    cl_mem ocl_buf = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(int), &data[0], &err);
    sycl::buffer<int, 1> buffer = sycl::make_buffer<sycl::backend::opencl, int>(ocl_buf, ctx);

    auto event1 = sycl_kerenel_add_2(buffer, size, q);

    auto event2 = sycl_kerenel_mul2(buffer, size, q);

    auto event3 = add_ocl_kernel_add_1(ocl_ctx, ocl_dev, ctx, buffer, size, q);

    // Test found: don't need to sync here.
    // event1.wait();
    // event2.wait();
    clEnqueueReadBuffer(ocl_queue, ocl_buf, CL_TRUE, 0, size * sizeof(int), &data[0], 0, NULL, NULL);

    for (int i = 0; i < size; i++)
    {
        if (data[i] != (i + 2) * 2 + 1)
        {
            std::cout << "Results did not validate at index " << i << "!\n";
            return -1;
        }
    }
    std::cout << "Success!\n";
    return 0;
}