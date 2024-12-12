#include <omp.h>
#include <cblas.h>

#include "private.hpp"
#include "my_common.hpp"

float matmal_kernel_ref(MMParamsInput::PTR input, MMParamsOutput::PTR output)
{
    const size_t &M = input->_m;
    const size_t &K = input->_k;
    const size_t &N = input->_n;

    const float *A = input->_a;
    const float *B = input->_b;
    float *C = output->_c;

    assert(M == output->_m);
    assert(K == output->_k);

    std::cout << "== Kernel:      " << __FUNCTION__ << std::endl;
    std::cout << "Input A         = " << M << " x " << K << std::endl;
    std::cout << "Input B         = " << K << " x " << N << std::endl;
    std::cout << "Output C        = " << M << " x " << N << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    // A: [M*K]; B: [K*N]; C: [M*N]

    // #pragma omp parallel for num_threads(32)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int p = 0; p < K; p++) {
                C[i * N + j] += A[i * K + p] * B[p * N + j];
                // C[j * M + i] += A[p * M + i] * B[j * K + p];
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_dur = tm_diff_ms(t1, t2);

    std::cout << "Ref Kernel Execution Time: " << cpu_dur << " ms\n"
              << std::endl;
    return cpu_dur;
}

float matmal_kernel_openblas(MMParamsInput::PTR input, MMParamsOutput::PTR output) {
    const size_t &m = input->_m;
    const size_t &k = input->_k;
    const size_t &n = input->_n;

    const float *A = input->_a;
    const float *B = input->_b;
    float *C = output->_c;

    auto t1 = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1, A, m, B, k, 0, C, m);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1, B, k, A, m, 0, C, m);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_dur = tm_diff_ms(t1, t2);
    return cpu_dur;
}