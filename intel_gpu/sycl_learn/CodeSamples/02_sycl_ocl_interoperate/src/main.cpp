
#include "private.hpp"

int main()
{
    std::cout << "== Debug MACRO tip:" << std::endl;
    std::cout << "  Default:        test accuracy only." << std::endl;
    std::cout << "  PERFORMANCE=1:  Test performance only." << std::endl;

    // test_sycl_olc_interoperate_l0_backend();
    // test_sycl_olc_interoperate_l0_backend_reoder_weights();
    // test_sycl_olc_interoperate_ocl_backend();
    test_sycl_olc_interoperate_l0_backend_rope_ref();
    return 0;
}