// Reference:
// https://www.intel.com/content/www/us/en/developer/articles/training/programming-data-parallel-c.html
// Write kernel 2 ways:
// 1: Sycl::buffer
// 2: Sycl 2020 USM(supports pointer-based memory management.)
// 3: Sycl 2020 TerseSyntax
// Examples reference:
// 1: https://github.com/jeffhammond/dpcpp-tutorial/blob/master/saxpy-usm.cc
// 2: https://github.com/jeffhammond/dpcpp-tutorial/blob/master/saxpy-usm2.cc
// 3: https://github.com/jeffhammond/dpcpp-tutorial/blob/master/saxpy-usm3.cc

#include "my_common.hpp"

const float xval(1);
const float yval(2);
const float zval(2);
const float aval(3);
const float correct = (zval + aval * xval + yval);

void fun_1_sycl_buffer()
{
    Start_Test();
    size_t length = 1024;

    std::vector<float> h_X(length, xval);
    std::vector<float> h_Y(length, yval);
    std::vector<float> h_Z(length, zval);
    std::vector<float> h_expected(length, correct);

    try
    {
        sycl::queue q(sycl::gpu_selector_v);
        print_device_beckend(q);
        const float A(aval);

        // sycl::buffer warp processing buffer.
        sycl::buffer<float, 1> d_X{h_X.data(), sycl::range<1>(h_X.size())};
        sycl::buffer<float, 1> d_Y{h_Y.data(), sycl::range<1>(h_Y.size())};
        sycl::buffer<float, 1> d_Z{h_Z.data(), sycl::range<1>(h_Z.size())};

        q.submit([&](sycl::handler &h)
                 {
            auto X = d_X.get_access<sycl::access::mode::read>(h);
            auto Y = d_Y.get_access<sycl::access::mode::read>(h);
            auto Z = d_Z.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for<class kernel_1_sycl_buffer>(sycl::range<1>(length), [=](sycl::id<1> it){
                const int i = it[0];
                Z[i] += A * X[i] + Y[i];
            }); });
        q.wait();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    auto r = check_result<float>(h_Z, h_expected, FLT_MIN);
    std::cout << " Result is " << (r ? "expected. " : "not expected.") << std::endl;
    End_Test();
}

void fun_2_usm()
{
    Start_Test();
    size_t length = 1024;

    sycl::queue q(sycl::gpu_selector_v);
    print_device_beckend(q);

    auto X = sycl::malloc_shared<float>(length, q);
    auto Y = sycl::malloc_shared<float>(length, q);
    auto Z = sycl::malloc_shared<float>(length, q);

    for (size_t i = 0; i < length; i++)
    {
        X[i] = xval;
        Y[i] = yval;
        Z[i] = zval;
    }

    try
    {
        const float A(aval);

        // USM: still based on pointer.
        q.submit([&](sycl::handler &h)
                 { h.parallel_for<class kernel_fun_2_usm>(sycl::range<1>{length}, [=](sycl::id<1> i)
                                                          { Z[i] += A * X[i] + Y[i]; }); });
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    std::vector<float> h_expected(length, correct);
    auto r = check_result<float>(Z, h_expected.data(), length, FLT_MIN);
    std::cout << " Result is " << (r ? "expected. " : "not expected.") << std::endl;
    End_Test();
}

void fun_3_terse_syntax()
{
    Start_Test();
    size_t length = 1024;

    // host data
    std::vector<float> h_X(length, xval);
    std::vector<float> h_Y(length, yval);
    std::vector<float> h_Z(length, zval);

    sycl::queue q(sycl::gpu_selector_v);
    print_device_beckend(q);

    auto X = sycl::malloc_device<float>(length, q);
    auto Y = sycl::malloc_device<float>(length, q);
    auto Z = sycl::malloc_device<float>(length, q);

    const size_t bytes = length * sizeof(float);

    q.memcpy(X, h_X.data(), bytes);
    q.memcpy(Y, h_Y.data(), bytes);
    q.memcpy(Z, h_Z.data(), bytes);
    q.wait();

    try
    {
        const float A(aval);

        // Terse Syntax
        // Means: no sycl::handler.
        q.parallel_for(sycl::range<1>{length}, [=](sycl::id<1> i)
                       { Z[i] += A * X[i] + Y[i]; });
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    q.memcpy(h_Z.data(), Z, bytes);
    q.wait();

    std::vector<float> h_expected(length, correct);
    auto r = check_result<float>(h_Z, h_expected, FLT_MIN);
    std::cout << " Result is " << (r ? "expected. " : "not expected.") << std::endl;
    End_Test();
}

int main(int argc, char *argv[])
{
    // sycl::buffer
    fun_1_sycl_buffer();

    // USM, still use pointer.
    fun_2_usm();

    // No sycl::handler
    fun_3_terse_syntax();
    return 0;
}