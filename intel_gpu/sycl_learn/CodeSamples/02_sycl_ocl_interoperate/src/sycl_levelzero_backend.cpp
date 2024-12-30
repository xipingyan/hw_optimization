#include "private.hpp"

#include <iostream>
#include <fstream>
#include <string>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

std::string load_kernel(std::string kernel_fn)
{
	std::ifstream kernel_file(kernel_fn.c_str(), std::ios::in | std::ios::binary);
	if (kernel_file.is_open())
	{
		std::string ret;
		auto beg = kernel_file.tellg();
		kernel_file.seekg(0, std::ios::end);
		auto end = kernel_file.tellg();
		kernel_file.seekg(0, std::ios::beg);

		ret.resize((size_t)(end - beg));
		kernel_file.read(&ret[0], (size_t)(end - beg));

		return {std::move(ret)};
	}
	return std::string();
}

sycl::event launchOpenCLKernelOnline(sycl::queue &q, std::string source,
									 std::string func_name, std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> &params,
									 size_t length,
									 sycl::event &dep_event)
{
	std::cout << "  == Start to kernel_bundle opencl source" << std::endl;
	sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
		syclex::create_kernel_bundle_from_source(
			q.get_context(),
			syclex::source_language::opencl,
			source);

	// Compile and link the kernel from the source definition.
	std::cout << "  == Start to kernel_bundle kb_src" << std::endl;
	sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
		syclex::build(kb_src);

	// Get a "kernel" object representing the kernel defined in the
	// source string.
	std::cout << "  == Start to get sycl::kernel" << std::endl;
	sycl::kernel k = kb_exe.ext_oneapi_get_kernel(func_name);

	// constexpr int N = length;
	constexpr int WGSIZE = 1;

	std::cout << "  == Start to submit" << std::endl;
	return q.submit([&](sycl::handler &cgh)
					{
                        cgh.depends_on(dep_event);

						for (int i = 0; i < params.size(); i++)
						{
							if (params[i].second)
							{
								sycl::accessor acc_param{params[i].first, cgh, sycl::read_write};
								cgh.set_arg(i, acc_param);
							}
							else
							{
								sycl::accessor acc_param{params[i].first, cgh, sycl::read_only};
								cgh.set_arg(i, acc_param);
							}
						}

						// Invoke the kernel over an nd-range.
                        // sycl::nd_range ndr{{length}, {WGSIZE}};
						sycl::nd_range<3> ndr{{384,2,1}, {384,2,1}};
                        cgh.parallel_for(ndr, k); });
}

void read_params(sycl::queue queue, std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> &params,
				 float **in_buf, sycl::half **out_buf, size_t &length)
{
	std::string param_fn_1 = "reorder_data_11511535514038671586_0_0__sa_6144.dat";
	std::string param_fn_2 = "reorder_data_11511535514038671586_0_0__sa_3072.dat";
	std::string param_fn_3 = "reorder_data_11511535514038671586_0_0__sa_64.dat";
	*in_buf = sycl::malloc_shared<float>(6144 / sizeof(float), queue);
	*out_buf = sycl::malloc_shared<sycl::half>(3072 / sizeof(sycl::half), queue);
	auto cell_buf = sycl::malloc_shared<int32_t>(64 / sizeof(int32_t), queue);

	FILE *pf1 = fopen(param_fn_1.c_str(), "rb");
	fread(*in_buf, sizeof(u_char), 6144, pf1);
	fclose(pf1);
	length = 6144 / sizeof(float);

	FILE *pf2 = fopen(param_fn_2.c_str(), "rb");
	fread(*out_buf, sizeof(u_char), 3072, pf2);
	fclose(pf2);

	FILE *pf3 = fopen(param_fn_3.c_str(), "rb");
	fread(cell_buf, sizeof(u_char), 64, pf1);
	fclose(pf3);

	sycl::buffer param_in1((uint8_t *)(*in_buf), sycl::range{6144});
	params.push_back({param_in1, false});
	sycl::buffer param_in2((uint8_t *)(*out_buf), sycl::range{3072});
	params.push_back({param_in2, true});
	sycl::buffer param_in3((uint8_t *)(cell_buf), sycl::range{64});
	params.push_back({param_in3, false});

	for (size_t i = 0; i < length; i++)
	{
		(*in_buf)[i] = i + 0.12345678f;
	}

	std::cout << "  == cell_buf = ";
	for (size_t i = 0; i < 64 / sizeof(int32_t); i++)
	{
		std::cout << cell_buf[i] << ", ";
	}
	std::cout << std::endl;
}

int test_sycl_olc_interoperate_l0_backend()
{
	std::cout << "== Test: " << __FUNCTION__ << ":" << __LINE__ << std::endl;
	std::string kernel_source = load_kernel("../02_sycl_ocl_interoperate/src/kernel_f32_to_f16.cl");
	// std::cout << "  kernel_source = " << kernel_source << std::endl;

	auto queue = sycl::queue(sycl::gpu_selector_v);
	std::cout << "  == Using "
			  << queue.get_device().get_info<sycl::info::device::name>()
			  << ", Backend: " << queue.get_backend()
			  << std::endl;

	sycl::event ev;
	std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> params;

#define READ_PARAM 0
#if !READ_PARAM
	size_t length = 6144 / sizeof(float);
	auto in_buf = sycl::malloc_shared<float>(length, queue);
	auto out_buf = sycl::malloc_shared<sycl::half>(length, queue);
	for (size_t i = 0; i < length; i++)
	{
		in_buf[i] = i + 0.12345678f;
	}
	auto cell_buf = sycl::malloc_shared<int32_t>(64 / sizeof(int32_t), queue);
	int32_t cell_arr[] = {1, 2, 1, 1, 1, 1, 384, 1, 1, 2, 1, 1, 1, 1, 384, 1};
	for (size_t i = 0; i < 64 / sizeof(int32_t); i++)
	{
		cell_buf[i] = cell_arr[i];
	}

	sycl::buffer param_in((uint8_t *)(in_buf), sycl::range{length * sizeof(float)});
	params.push_back({param_in, false});
	sycl::buffer param_out((uint8_t *)(out_buf), sycl::range{length * sizeof(sycl::half)});
	params.push_back({param_out, true});
	sycl::buffer param_cell((uint8_t *)(cell_buf), sycl::range{64});
	params.push_back({param_cell, false});
#else
	size_t length = 1;
	float *in_buf = nullptr;
	sycl::half *out_buf = nullptr;
	read_params(queue, params, &in_buf, &out_buf, length);
#endif

	launchOpenCLKernelOnline(queue, kernel_source, "reorder_data_11511535514038671586_0_0__sa", params, length, ev);

	// Print IN/OUT
	std::cout << "  == Compare input and output:" << std::endl;
	for (size_t i = 0; i < 5; i++)
	{
		std::cout << "    in_buf[" << i << "] = " << in_buf[i] << " VS out_buf[" << i << "] = " << out_buf[i] << std::endl;
	}

	std::cout << std::endl;
	return 0;
}