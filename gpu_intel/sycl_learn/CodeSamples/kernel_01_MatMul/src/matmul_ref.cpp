#include <omp.h>
#include <cblas.h>

#include "private.hpp"
#include "my_common.hpp"

// Refer:https://salykova.github.io/matmul-cpu
float matmal_kernel_ref(MMParamsInput<float>::PTR input, MMParamsOutput<float>::PTR output)
{
    const size_t &M = input->_m;
    const size_t &K = input->_k;
    const size_t &N = input->_n;

    const float *A = input->_a;
    const float *B = input->_b;
    float *C = output->_c;

    assert(M == output->_m);
    assert(N == output->_n);

    std::cout << "== Kernel: " << __FUNCTION__ << std::endl;
    std::cout << "  Input A = " << M << " x " << K << std::endl;
    std::cout << "  Input B = " << K << " x " << N << std::endl;
    std::cout << "  Output C= " << M << " x " << N << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int p = 0; p < K; p++)
            {
                // Row order:
                C[i * N + j] += A[i * K + p] * B[p * N + j];

                // Colum-major order:
                // C[j * M + i] += A[p * M + i] * B[j * K + p];
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dur = tm_diff_ms(t1, t2);

    std::cout << "  Ref Kernel Execution Time: " << dur << " ms\n"
              << std::endl;
    return dur;
}

float matmal_kernel_openblas(MMParamsInput<float>::PTR input, MMParamsOutput<float>::PTR output)
{
    const size_t &m = input->_m;
    const size_t &k = input->_k;
    const size_t &n = input->_n;

    const float *A = input->_a;
    const float *B = input->_b;
    float *C = output->_c;

    std::cout << "== Kernel: " << __FUNCTION__ << std::endl;
    std::cout << "  Input A = [" << m << " x " << k << "]; B = " << k << " x " << n << "]" << std::endl;
    std::cout << "  Output C= " << m << " x " << n << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    // lda,ldb,ldc are their stride.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dur = tm_diff_ms(t1, t2);
    std::cout << "  Time = " << dur << " ms \n\n";
    return dur;
}