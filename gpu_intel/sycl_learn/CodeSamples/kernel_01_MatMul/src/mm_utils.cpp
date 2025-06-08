#include "private.hpp"

MMParamsInput<sycl::half>::PTR cvt_f32_to_half(MMParamsInput<float>::PTR ptr)
{
    const auto &M = ptr->_m;
    const auto &N = ptr->_n;
    const auto &K = ptr->_k;
    MMParamsInput<sycl::half>::PTR ptrHalf = MMParamsInput<sycl::half>::create(M, K, N);

    for (size_t m = 0; m < M; m++)
    {
        for (size_t k = 0; k < K; k++)
        {
            ptrHalf->_a[m * K + k] = static_cast<sycl::half>(ptr->_a[m * K + k]);
        }
    }
    for (size_t k = 0; k < K; k++)
    {
        for (size_t n = 0; n < N; n++)
        {
            ptrHalf->_b[k * N + n] = static_cast<sycl::half>(ptr->_b[k * N + n]);
        }
    }
    return ptrHalf;
}

bool is_same(std::string prefix, MMParamsOutput<float>::PTR output1, MMParamsOutput<float>::PTR output2, float T, bool trans_b) {
    assert(output1->_m == output2->_m);
    assert(output1->_n == output2->_n);

    bool bsame = true;
    for (size_t i = 0; i < output1->_m; i++)
    {
        for (size_t j = 0; j < output2->_n; j++) {
            const auto& a_val = output1->_c[i * output1->_n + j];
            const auto& b_val = trans_b ? output2->_c[j * output1->_n + i] : output2->_c[i * output1->_n + j];

            if (fabs(a_val - b_val) > T)
            {
                std::cout << prefix << ", [" << i << ", " << j << "], diff = " << fabs(a_val - b_val) << ", output1 = " << a_val << ", output2 = " << b_val << std::endl;
                bsame = false;
                return false;
            }
        }
    }

    if (bsame)
        std::cout << prefix << ", is same." << std::endl;
    return true;
}