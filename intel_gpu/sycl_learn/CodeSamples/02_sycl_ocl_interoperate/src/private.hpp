#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <sycl/sycl.hpp>

int test_sycl_olc_interoperate_l0_backend();
int test_sycl_olc_interoperate_l0_backend_reoder_weights();
int test_sycl_olc_interoperate_ocl_backend();

// Just verify if opencl and sycl have light different result.
int test_sycl_olc_interoperate_l0_backend_rope_ref();

struct DumpData
{
    std::string format;
    std::vector<int> shape;
    std::vector<float> data;
    std::string to_string();
    sycl::half* to_half(sycl::queue queue);
    int* to_int(sycl::queue queue);
};
DumpData load_dump_data(std::string fn);

bool check_path_exist(const std::string& path);

sycl::event launchOpenCLKernelOnlineLevelZero(sycl::queue &q, std::string source,
                                              std::string func_name, std::vector<void *> &params,
                                              sycl::event &dep_event, bool test_performance);