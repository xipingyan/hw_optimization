// Reference:
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-1/use-cmake-with-the-compiler.html

#include <iostream>
#include <sycl/sycl.hpp>
#include <cmath>

int main(int argc, char* argv[])
{
    sycl::queue queue;

    std::cout << "Using "
        << queue.get_device().get_info<sycl::info::device::name>()
        << std::endl;

    // Compute the first n_items values in a well known sequence
    constexpr int n_items = 16;
    int *items = sycl::malloc_shared<int>(n_items, queue);
    queue.parallel_for(sycl::range<1>(n_items), [items] (sycl::id<1> i) {
        float x1 = powf((1.0f + sqrtf(5.0))/2, i);
        float x2 = powf((1.0f - sqrtf(5.0))/2, i);
        items[i] = roundf((x1 - x2)/sqrtf(5));
    }).wait();

    for(int i = 0 ; i < n_items ; ++i) {
        std::cout << items[i] << std::endl;
    }
    free(items, queue);

    return 0;
}