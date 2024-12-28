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
	std::string func_name, std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>>& params, 
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
                        sycl::nd_range ndr{{length}, {WGSIZE}};
                        cgh.parallel_for(ndr, k);
                    });
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

	size_t length = 10;
	auto in_buf = sycl::malloc_shared<float>(length, queue);
	auto out_buf = sycl::malloc_shared<sycl::half>(length, queue);
	for (size_t i = 0; i < length; i++)
	{
		in_buf[i] = i + 0.12345678f;
	}

	std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> params;

	sycl::buffer param_in((uint8_t *)(in_buf), sycl::range{length * sizeof(float)});
	params.push_back({param_in, false});
	sycl::buffer param_out((uint8_t *)(out_buf), sycl::range{length * sizeof(sycl::half)});
	params.push_back({param_out, true});

	launchOpenCLKernelOnline(queue, kernel_source, "reorder_data_11511535514038671586_0_0__sa", params, length, ev);

	// Print IN/OUT
	std::cout << "  == Compare input and output:" << std::endl;
	for (size_t i = 0; i < length; i++)
	{
		std::cout << "    in_buf[" << i << "] = " << in_buf[i] << " VS out_buf[" << i << "] = " << out_buf[i] << std::endl;
	}

	std::cout << std::endl;
	return 0;
}