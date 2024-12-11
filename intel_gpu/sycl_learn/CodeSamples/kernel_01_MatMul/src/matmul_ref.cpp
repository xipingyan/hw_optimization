#include <omp.h>

#include "my_sycl_kernel.hpp"

float matmal_kernel_ref(MMParamsInput::PTR input, MMParamsOutput::PTR output)
{
    const size_t &M = input->_m;
    const size_t &N = input->_n;
    const size_t &K = input->_k;

    assert(M == output->_m);
    assert(K == output->_k);

    std::cout << "== Kernel:      " << __FUNCTION__ << std::endl;
    std::cout << "Input A         = " << M << " x " << N << std::endl;
    std::cout << "Input B         = " << N << " x " << K << std::endl;
    std::cout << "Output C        = " << M << " x " << K << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    // #pragma omp parallel for num_threads(32)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float tmp = 0;
            for (int n = 0; n < N; n ++) {
                tmp += input->_a[i * N + n] * input->_b[n * K + j];
            }
            output->_c[i * K + j] = tmp;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Ref Kernel Execution Time: " << cpu_dur << " ms\n"
              << std::endl;
    return cpu_dur;
}