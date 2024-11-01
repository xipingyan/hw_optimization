// Reference:
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-1/use-cmake-with-the-compiler.html

#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

std::vector<float> vec_add(std::vector<float>& a, std::vector<float>& b, sycl::queue& queue) {
    std::vector<float> sum(a.size());
    // Compute the first n_items values in a well known sequence
    size_t n_items = a.size();
    float *items = sycl::malloc_shared<float>(n_items, queue);
    float *a_dev = sycl::malloc_shared<float>(n_items, queue);
    float *b_dev = sycl::malloc_shared<float>(n_items, queue);
    for (size_t i = 0; i < a.size(); i++) {
        a_dev[i] = a[i];
        b_dev[i] = b[i];
    }

    queue.parallel_for(sycl::range<1>(n_items), [items, a_dev, b_dev] (sycl::id<1> i) {
        float x1 = a_dev[i];
        float x2 = b_dev[i];
        // items[i] = roundf(x1+x2);
        items[i] = x1 + x2;
    }).wait();

    for (size_t i = 0; i < a.size(); i++) {
            sum[i] = items[i];
    }
    free(items, queue);
    free(a_dev, queue);
    free(b_dev, queue);
    return sum;
}

// std::vector<float> vec_div_4(std::vector<float>& a, sycl::queue& queue) {
//     std::vector<float> dst(a.size());
//     // Compute the first n_items values in a well known sequence
//     size_t n_items = a.size();
//     float *items = sycl::malloc_shared<float>(n_items, queue);
//     float *a_dev = sycl::malloc_shared<float>(n_items, queue);
//     for (size_t i = 0; i < a.size(); i++) {
//         a_dev[i] = a[i];
//     }

//     queue.parallel_for(sycl::range<1>(n_items), [items, a_dev] (sycl::id<1> i) {
//         items[i] = a_dev[i] / 4.f;
//     }).wait();

//     for (size_t i = 0; i < a.size(); i++) {
//             dst[i] = items[i];
//     }
//     free(items, queue);
//     free(a_dev, queue);
//     return dst;
// }

int main(int argc, char* argv[])
{
    sycl::queue queue{sycl::gpu_selector_v};
    // sycl::queue queue;
    std::cout << "Using "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    std::vector<float> a, b, expected;
    for (int i = 0; i < 10000; i++) {
        a.push_back(i);
        b.push_back(i);
        expected.push_back(i+i);
        // expected.push_back((i+i)/4.f);
    }
    auto result = vec_add(a, b, queue);
    // result = vec_div_4(result, queue);

    bool result_is_expected = true;
    for (int i = 0; i < 10000; i++) {
        if (fabs(expected[i] - result[i]) > 0.0001f) {
            std::cout << "== Result [" << i << "] diff: " << fabs(expected[i] - result[i]) << ", result=" << result[i] <<  std::endl;
            result_is_expected = false;
        }
    }
    std::cout << "Done, result is " << (result_is_expected ? "expected" : "not expected") << std::endl;
    return 0;
}