#include "my_sycl_kernel.hpp"

bool is_same(MMParamsOutput::PTR output1, MMParamsOutput::PTR output2, float T) {
    assert(output1->_m == output2->_m);
    assert(output1->_k == output2->_k);

    bool bsame = true;
    for (size_t i = 0; i < output1->_m; i++)
    {
        for (size_t j = 0; j < output2->_k; j++) {
            if (fabs(output1->_c[i * output1->_k + j] - output2->_c[i * output1->_k + j]) > T) {
                std::cout << "  == diff: [" << i << ", " << j << "], diff = " << fabs(output1->_c[i * output1->_k + j] - output2->_c[i * output1->_k + j]) << ", output1 = " << output1->_c[i * output1->_k + j] << ", output2 = " << output2->_c[i * output1->_k + j] << std::endl;
                bsame = false;
            }
        }
    }
    return true;
}