#include "private.hpp"

bool is_same(MMParamsOutput::PTR output1, MMParamsOutput::PTR output2, float T) {
    assert(output1->_m == output2->_m);
    assert(output1->_n == output2->_n);

    bool bsame = true;
    for (size_t i = 0; i < output1->_m; i++)
    {
        for (size_t j = 0; j < output2->_n; j++) {
            if (fabs(output1->_c[i * output1->_n + j] - output2->_c[i * output1->_n + j]) > T) {
                std::cout << "  == diff: [" << i << ", " << j << "], diff = " << fabs(output1->_c[i * output1->_n + j] - output2->_c[i * output1->_n + j]) << ", output1 = " << output1->_c[i * output1->_n + j] << ", output2 = " << output2->_c[i * output1->_n + j] << std::endl;
                bsame = false;
                return false;
            }
        }
    }

    return true;
}