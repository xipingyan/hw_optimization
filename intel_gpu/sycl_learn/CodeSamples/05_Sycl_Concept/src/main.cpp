// Reference:
// https://enccs.github.io/sycl-workshop/expressing-parallelism-nd-range/
// Some important concepts
// Work-item : 1 element
// Sub-group : contiguous one group elements
// Work-group: Parts of ND-Range
// ND-Range  : 3 dim's Tensor.

// sycl::nd_range, params:
// globalSize :
// localSize  :
// offset     : id
#include <random>

#include "my_common.hpp"

int main(int argc, char *argv[])
{
    // set up queue on any available device
    sycl::queue queue(sycl::gpu_selector_v);
    print_device_beckend(queue, "");

    // Function: Cij = Aik * Bkj

    // initialize input and output memory on the host
    constexpr size_t N = 8;
    std::vector<float> a(N * N), b(N * N), c(N * N);

    // fill a and b with random numbers in the unit interval
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::generate(a.begin(), a.end(), [&dist, &mt]()
                  { return dist(mt); });
    std::generate(b.begin(), b.end(), [&dist, &mt]()
                  { return dist(mt); });

    // zero-out c
    std::fill(c.begin(), c.end(), 0.0f);

    {
        // size of local range
        constexpr size_t B = 4;
        // Create buffers associated with inputs and output
        // a_buf=b_buf=c_buf=Mat(N,N)
        sycl::buffer<float, 2> a_buf(a.data(), sycl::range<2>(N, N)),
            b_buf(b.data(), sycl::range<2>(N, N)), c_buf(c.data(), sycl::range<2>(N, N));

        // Submit the kernel to the queue
        queue.submit([&](sycl::handler &cgh)
                     {
                        sycl::accessor a { a_buf, cgh };
                        sycl::accessor b { b_buf, cgh };
                        sycl::accessor c { c_buf, cgh };

                        // declere global and local ranges
                        sycl::range global { N, N };
                        sycl::range local { B, B };

                        // Setup sycl::stream class to print standard output from the device code
                        // Suitable buffer size can make sure print all data.
                        auto out = sycl::stream((N*N)*1024, 768, cgh);
                        cgh.parallel_for(sycl::nd_range { global, local }, [=](sycl::nd_item<2> it) {
                            auto i = it.get_global_id(0); // dim order: [0,1]
                            auto j = it.get_global_id(1);
                            auto g_linear_id = it.get_global_linear_id();
                            auto local_id = it.get_local_id();
                            auto nd = it.get_nd_range();

                            out << "i, j = " << i << ", " << j << ", id(i*N+j) = " << i * N + j << ", g_linear_id=" << g_linear_id << ", local_id=" << local_id << ",nd=" << nd << sycl::endl;

                            for (auto k = 0; k < N; ++k)
                            {
                                c[j][i] += a[j][k] * b[k][i];
                            }
                        }); });
        queue.wait();
    }

    // Check that all outputs match serial execution
    bool passed = true;
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            float gold = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                gold += a[j * N + k] * b[k * N + i];
            }
            // GPU and CPU's accuracy has a little diff.
            if (std::fabs(gold - c[j * N + i]) > 1.0e-04)
            {
                passed = false;
                std::cout << " == diff: " << std::fabs(gold - c[j * N + i]) << std::endl;
            }
        }
    }
    std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
    return (passed) ? 0 : 1;
}