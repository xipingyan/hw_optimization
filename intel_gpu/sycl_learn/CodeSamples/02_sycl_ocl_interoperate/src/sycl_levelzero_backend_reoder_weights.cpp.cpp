#include "private.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

static std::string load_kernel(std::string kernel_fn)
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

static sycl::event launchOpenCLKernelOnline(sycl::queue &q, std::string source,
											std::string func_name, std::vector<std::pair<sycl::buffer<uint8_t, 1>, bool>> &params,
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
						sycl::nd_range<3> ndr{{192,384,1}, {192,4,1}};
                        cgh.parallel_for(ndr, k); });
}

int test_sycl_olc_interoperate_l0_backend_reoder_weights()
{
	std::cout << "== Test: " << __FUNCTION__ << ":" << __LINE__ << std::endl;
	std::string kernel_source = load_kernel("../02_sycl_ocl_interoperate/src/reorder_weights_7_weight_0_0.cl");
	// std::cout << "  kernel_source = " << kernel_source << std::endl;

	auto queue = sycl::queue(sycl::gpu_selector_v);
	std::cout << "  == Using "
			  << queue.get_device().get_info<sycl::info::device::name>()
			  << ", Backend: " << queue.get_backend()
			  << std::endl;

	sycl::event ev;
	std::vector<std::pair<sycl::buffer<uint8_t, 1>, bool>> params;

	size_t length = 192 * 384;
	auto in_buf = sycl::malloc_shared<float>(length, queue);
	auto out_buf = sycl::malloc_shared<sycl::half>(length, queue);
	for (size_t i = 0; i < length; i++)
	{
		in_buf[i] = i % 10 + 0.12345678f;
	}

	sycl::buffer param_in((uint8_t *)(in_buf), sycl::range{length * sizeof(float)});
	params.push_back({param_in, false});
	sycl::buffer param_out((uint8_t *)(out_buf), sycl::range{length * sizeof(sycl::half)});
	params.push_back({param_out, true});

	auto ret_ev = launchOpenCLKernelOnline(queue, kernel_source, "reorder_weights_7_weight_0_0", params, length, ev);
	ret_ev.wait();

	// Print IN/OUT
	std::cout << "  == Compare input and output:" << std::endl;
	int diff_num = 0;
	for (size_t i = 0; i < length; i++)
	{
		if (fabs(in_buf[i] - out_buf[i]) > 0.01f)
		{
			diff_num++;
			std::cout << "    in_buf[" << i << "] = " << in_buf[i] << " VS out_buf[" << i << "] = " << out_buf[i] << std::endl;
		}
		if (diff_num > 5)
		{
			break;
		}
	}

	std::cout << std::endl;
	std::cout << "  == Input and Ouput are " << (diff_num == 0 ? "Same." : "Not Same.") << std::endl;
	return 0;
}