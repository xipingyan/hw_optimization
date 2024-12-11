#include "private.hpp"

// Version:01, My original implementation.
// Reference: https://github.com/intel/llvm/issues/8163

float matmal_kernel_1(sycl::queue &q, MMParamsInput::PTR input, MMParamsOutput::PTR output, int group_x, int group_y)
{
    const size_t &M = input->_m;
    const size_t &K = input->_k;
    const size_t &N = input->_n;

    std::cout << "== Kernel:      " << __FUNCTION__ << std::endl;
    std::cout << "Input A         = " << M << " x " << K << std::endl;
    std::cout << "Input B         = " << K << " x " << N << std::endl;
    std::cout << "Output C        = " << M << " x " << N << std::endl;
    std::cout << "WORK_GROUP_SIZE = " << group_x << " x " << group_y << std::endl;

    assert(M == output->_m);
    assert(N == output->_n);

    // sycl::buffer a(input->_a), b(input->_b), c(input->_c);
    sycl::buffer a(input->_a, sycl::range{M * K});
    sycl::buffer b(input->_b, sycl::range{K * N});
    sycl::buffer c(output->_c, sycl::range{M * N});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto e = q.submit([&](sycl::handler &h)
                      {
        auto A = a.get_access<sycl::access::mode::read>(h);
        auto B = b.get_access<sycl::access::mode::read>(h);
        auto C = c.get_access<sycl::access::mode::read_write>(h);

        sycl::range<2> global_size(M, N), wg_size(group_x, group_y);

        h.parallel_for<class kernel_parallel_for>(sycl::nd_range<2>(global_size, wg_size), [=](sycl::nd_item<2> item)
                                                  {
                                                    int i = item.get_global_id(0), j = item.get_global_id(1);
                                                    for (int k = 0; k < K; k++){
                                                        C[i * N + j] += A[i * K + k] * B[k * N + j];}
                                                    }); });
    e.wait();
    auto t2 = std::chrono::high_resolution_clock::now();

    auto kernel_duration =
        (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
         e.get_profiling_info<sycl::info::event_profiling::command_start>());
    auto tm = kernel_duration / 1e+6;

    auto cpu_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Kernel Execution Time: " << tm << " ms, cpu time: "<< cpu_dur <<" ms\n" << std::endl;
    return tm;
}