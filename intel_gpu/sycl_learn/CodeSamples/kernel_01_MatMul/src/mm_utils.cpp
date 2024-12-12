#include "private.hpp"

bool is_same(std::string prefix, MMParamsOutput::PTR output1, MMParamsOutput::PTR output2, float T, bool trans_b) {
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
                // return false;
            }
        }
    }

    std::cout << prefix << ", is same." << std::endl;
    return true;
}