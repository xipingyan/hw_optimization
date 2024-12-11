// Reference:
// https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/executing-multiple-kernels-on-the-device-at-the.html

#include <random>
#include <thread>

#include "my_common.hpp"
#include "my_sycl_kernel.hpp"

static bool bCmpAcc = std::getenv("CMP_ACC");

void test_matmul_01(sycl::queue queue)
{
    std::cout << "==========================" << std::endl;
    std::cout << "== Start: " << __FUNCTION__ << std::endl;
    // std::vector<size_t> mm_szs = {512, 1024, 2048, 4096};
    std::vector<size_t> mm_szs = {512};
    std::vector<float> durations;
    for (auto mm_sz : mm_szs)
    {
        auto inputs = MMParamsInput::create(512, 256, 512);
        auto output_ref = MMParamsOutput::create(512, 512);
        auto output = MMParamsOutput::create(512, 512);
        int group_x = 16;
        int group_y = 16;

        if (bCmpAcc)
            matmal_kernel_ref(inputs, output_ref);

        float min_tm = std::numeric_limits<float>::max();
        for (auto i = 0; i < 1; i++)
        {
            min_tm = fmin(min_tm, matmal_kernel_1(queue, inputs, output, group_x, group_y));
            if (bCmpAcc && i == 0)
            {
                std::cout << "== Compare with ref" << std::endl;
                auto r = is_same(output, output_ref);
                std::cout << "  is same = " << r << std::endl;
            }
        }
        durations.push_back(min_tm);
    }

    std::cout << "=================================" << std::endl;
    for (size_t i = 0; i < mm_szs.size(); i++)
    {
        std::cout << "== matmul size = " << mm_szs[i] << ", group size = 16, min tm = " << durations[i] << " ms" << std::endl;
    }
}

int main()
{
    std::cout << "**********************************" << std::endl;
    std::cout << " Compare accuracy with reference result." << std::endl;
    std::cout << " $ export CMP_ACC=1" << std::endl;
    std::cout << "**********************************" << std::endl;

    sycl::queue queue(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}});

    DEBUG_LOG << "Using "
              << queue.get_device().get_info<sycl::info::device::name>()
              << ", Backend: " << queue.get_backend()
              << std::endl;

    DEBUG_LOG << "== prepare input." << std::endl;

    test_matmul_01(queue);

    return 0;
}