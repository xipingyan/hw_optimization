// Reference:
// https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/executing-multiple-kernels-on-the-device-at-the.html

#include <random>
#include <thread>

#include "my_common.hpp"
#include "private.hpp"

static bool bCmpAcc = std::getenv("ACC");

void test_matmul_01(sycl::queue queue)
{
    std::cout << "==========================" << std::endl;
    std::cout << "== Start: " << __FUNCTION__ << std::endl;
    // std::vector<size_t> mm_szs = {512, 1024, 2048, 4096};
    std::vector<size_t> mm_szs = {4096};
    std::vector<std::pair<float,float>> durations;
    int group_x = 16;
    int group_y = 16;
    int iter = 8;

    for (auto mm_sz : mm_szs)
    {
        auto inputs = MMParamsInput<float>::create(mm_sz, 4096, mm_sz);
        auto inputs_f16 = cvt_f32_to_half(inputs);
        auto output_ref = MMParamsOutput<float>::create(mm_sz, mm_sz);
        auto output_cpu = MMParamsOutput<float>::create(mm_sz, mm_sz);
        auto output_openblas = MMParamsOutput<float>::create(mm_sz, mm_sz);
        auto output_sycl_gpu_1 = MMParamsOutput<float>::create(mm_sz, mm_sz);
        auto output_sycl_gpu_2 = MMParamsOutput<float>::create(mm_sz, mm_sz);

        if (bCmpAcc) {
            // matmal_kernel_ref(inputs, output_ref);
            matmal_kernel_openblas(inputs, output_openblas);
        }

        sycl::queue queue_cpu(sycl::cpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}});
        matmal_kernel_1(queue_cpu, inputs, output_cpu, group_x, group_y);

        sycl::queue queue_gpu(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}});

        float min_tm_f32 = std::numeric_limits<float>::max();
        float min_tm_f16 = std::numeric_limits<float>::max();
        for (auto i = 0; i < 5; i++)
        {
            min_tm_f32 = fmin(min_tm_f32, matmal_kernel_1(queue_gpu, inputs, output_sycl_gpu_1, group_x, group_y));
            min_tm_f16 = fmin(min_tm_f16, matmal_kernel_1_inp_f16(queue_gpu, inputs_f16, output_sycl_gpu_2, group_x, group_y));

            if (bCmpAcc && i == 0)
            {
                std::cout << "==========================================" << std::endl;
                inputs->print();
                std::cout << "== Compare with ref" << std::endl;
                // is_same("  gpu_vs_ref", output_sycl_gpu_1, output_ref, 1.0e-04);
                // is_same("  cpu_vs_ref", output_cpu, output_ref);
                // is_same("  cpu_vs_gpu", output_cpu, output_sycl_gpu_1);
                // is_same("  ref_vs_openblas", output_ref, output_openblas, 1.0e-04);
                is_same("  gpu_f32_vs_openblas", output_sycl_gpu_1, output_openblas, 1.0e-02);
                is_same("  gpu_f16_vs_openblas", output_sycl_gpu_2, output_openblas, 1.0e-01);
            }
        }
        durations.push_back({min_tm_f32, min_tm_f16});
    }

    std::cout << "=================================" << std::endl;
    std::cout << "== GPU loop " << iter << " , min infer time: " << std::endl;
    for (size_t i = 0; i < mm_szs.size(); i++)
    {
        std::cout << "== matmul size = " << mm_szs[i] << ", group size = 16, min tm_f32 = " << durations[i].first << " ms, min tm_f16 = " << durations[i].second << " ms" << std::endl;
    }
}

// Verify accuracy, sycl gpu vs cpu
// Conclusion:
// ref_f32_vs_gpu_f32 is same. 
// ref_f16_vs_gpu_f16 is same. 
// ref_f32_vs_gpu_f16 is diff, a = 504.624, b = 505, diff = 0.375671
void test_add(sycl::queue queue) {
    std::cout << "==========================" << std::endl;
    std::cout << "== Start: " << __FUNCTION__ << std::endl;

    int len = 1024;
    float *arr = new float[len];
    sycl::half *arr_fp16 = new sycl::half[len];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1); // uniform distribution between 0 and 1
    for (size_t i = 0; i < len; i++)
    {
        arr[i] = dis(gen);
        arr_fp16[i] = arr[i];
    }

    // Referenece
    float rslt_ref = 0;
    sycl::half rslt_ref_fp16 = 0;
    for (size_t i = 0; i < len; i++)
    {
        rslt_ref += arr[i];
        rslt_ref_fp16 += arr_fp16[i];
    }

    float rslt_gpu = 0;
    add_kernel_1(queue, arr, len, rslt_gpu, 16);
    sycl::half rslt_gpu_f16 = 0;
    add_kernel_1_f16(queue, arr_fp16, len, rslt_gpu_f16, 16);

    float ref_vs_gpu = fabs(rslt_ref - rslt_gpu);
    float ref_vs_gpuf16 = fabs(rslt_ref_fp16 - rslt_gpu_f16);

    auto cmp = [](std::string prefix, float a, float b)
    {
        float diff = fabs(a - b);
        if (diff > 0)
        {
            std::cout << prefix << " is diff, a = " << a << ", b = " << b << ", diff = " << diff << std::endl;
        }
        else
        {
            std::cout << prefix << " is same. " << std::endl;
        }
    };

    cmp("ref_f32_vs_gpu_f32", rslt_ref, rslt_gpu);
    cmp("ref_f16_vs_gpu_f16", rslt_ref_fp16, rslt_gpu_f16);
    cmp("ref_f32_vs_gpu_f16", rslt_ref, rslt_gpu_f16);
}

int main()
{
    std::cout << "**********************************" << std::endl;
    std::cout << " Compare accuracy with reference result." << std::endl;
    std::cout << " $ export ACC=1" << std::endl;
    std::cout << "**********************************" << std::endl;

    sycl::queue queue(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}});

    DEBUG_LOG << "Using "
              << queue.get_device().get_info<sycl::info::device::name>()
              << ", Backend: " << queue.get_backend()
              << std::endl;

    DEBUG_LOG << "== prepare input." << std::endl;

    test_matmul_01(queue);
    // test_add(queue);

    return 0;
}