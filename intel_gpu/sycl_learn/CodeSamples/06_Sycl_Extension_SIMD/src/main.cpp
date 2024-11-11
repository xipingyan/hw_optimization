// Reference:
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/explicit-simd-sycl-extension.html
// Just explicit call SIMD SYCL extension.
// Refer code from: https://github.com/intel/llvm-test-suite/blob/intel/SYCL/ESIMD/vadd_usm.cpp

#include <sycl/ext/intel/esimd.hpp>
#include "my_common.hpp"

constexpr size_t Size = 1024 * 1024 * 1024 * 5u;
constexpr unsigned VL = 32;
constexpr unsigned GroupSize = 8;

void vec_add_sycl()
{
    Start_Test();

    //   sycl::queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
    sycl::queue q(sycl::gpu_selector_v);
    print_device_beckend(q);

    float *A = sycl::malloc_shared<float>(Size, q);
    float *B = sycl::malloc_shared<float>(Size, q);
    float *C = sycl::malloc_shared<float>(Size, q);
    for (unsigned i = 0; i < Size; ++i)
    {
        A[i] = B[i] = i % 1024;
    }

    for (size_t l = 0; l < 5; l++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        try
        {
            auto e = q.submit([&](sycl::handler &cgh)
                              { cgh.parallel_for<class vec_add_sycl>(sycl::range<1>{Size},
                                                                     [=](sycl::id<1> i)
                                                                     {
                                                                         C[i] = A[i] + B[i];
                                                                     }); });
            e.wait();
        }
        catch (sycl::exception const &e)
        {
            std::cout << "SYCL exception caught: " << e.what() << '\n';
            free(A, q);
            free(B, q);
            free(C, q);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>((t2 - t1)).count();
        std::cout << " == loop: " << l << " time " << diff << " ms " << std::endl;
    }

    bool result_is_expected = true;
    for (size_t i = 0; i < Size; i++)
    {
        if (fabs(C[i] - (A[i] + B[i])) > FLT_MIN)
        {
            std::cout << "== Result [" << i << "] diff: " << fabs((A[i] + B[i]) - C[i]) << ", result=" << C[i] << std::endl;
            result_is_expected = false;
        }
    }
    std::cout << " Result is " << (result_is_expected ? "expected. " : "not expected.") << std::endl;
    free(A, q);
    free(B, q);
    free(C, q);
    End_Test();
}

void vec_add_sycl_simd()
{
    Start_Test();

    //   sycl::queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
    sycl::queue q(sycl::gpu_selector_v);
    print_device_beckend(q);

    float *A = sycl::malloc_shared<float>(Size, q);
    float *B = sycl::malloc_shared<float>(Size, q);
    float *C = sycl::malloc_shared<float>(Size, q);
    for (unsigned i = 0; i < Size; ++i)
    {
        A[i] = B[i] = i % 1024;
    }

    // We need that many workitems. Each processes VL elements of data.
    sycl::range<1> GlobalRange{Size / VL};
    // Number of workitems in each workgroup.
    sycl::range<1> LocalRange{GroupSize};

    sycl::nd_range<1> Range(GlobalRange, LocalRange);
    for (size_t l = 0; l < 5; l++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        try
        {
            auto e = q.submit([&](sycl::handler &cgh)
                              { cgh.parallel_for<class vec_add_sycl_simd>(Range,
                                                                          [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL
                                                                          {
                                                                              using namespace sycl::ext::intel::esimd;

                                                                              int i = ndi.get_global_id(0);
                                                                              simd<float, VL> va;
                                                                              va.copy_from(A + i * VL);
                                                                              simd<float, VL> vb;
                                                                              vb.copy_from(B + i * VL);
                                                                              simd<float, VL> vc = va + vb;
                                                                              vc.copy_to(C + i * VL);
                                                                          }); });
            e.wait();
        }
        catch (sycl::exception const &e)
        {
            std::cout << "SYCL exception caught: " << e.what() << '\n';
            free(A, q);
            free(B, q);
            free(C, q);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>((t2 - t1)).count();
        std::cout << " == loop: " << l << " time " << diff << " ms " << std::endl;
    }
    bool result_is_expected = true;
    for (size_t i = 0; i < Size; i++)
    {
        if (fabs(C[i] - (A[i] + B[i])) > FLT_MIN)
        {
            std::cout << "== Result [" << i << "] diff: " << fabs((A[i] + B[i]) - C[i]) << ", result=" << C[i] << std::endl;
            result_is_expected = false;
        }
    }
    std::cout << " Result is " << (result_is_expected ? "expected. " : "not expected.") << std::endl;
    free(A, q);
    free(B, q);
    free(C, q);
    End_Test();
}

int main(int argc, char *argv[])
{
    vec_add_sycl();
    vec_add_sycl_simd();
    // The both methold have same performance.
    // Output:
    // == Start test: vec_add_sycl
    // Using Intel(R) Arc(TM) A770 Graphics, Backend: ext_oneapi_level_zero
    // == loop: 0 time 937 ms 
    // == loop: 1 time 30 ms 
    // == loop: 2 time 30 ms 
    // == loop: 3 time 30 ms 
    // == loop: 4 time 30 ms 
    // Result is expected. 
    // == Finish test: vec_add_sycl
    // == Start test: vec_add_sycl_simd
    // Using Intel(R) Arc(TM) A770 Graphics, Backend: ext_oneapi_level_zero
    // == loop: 0 time 960 ms 
    // == loop: 1 time 30 ms 
    // == loop: 2 time 30 ms 
    // == loop: 3 time 30 ms 
    // == loop: 4 time 30 ms 
    // Result is expected. 
    return 0;
}