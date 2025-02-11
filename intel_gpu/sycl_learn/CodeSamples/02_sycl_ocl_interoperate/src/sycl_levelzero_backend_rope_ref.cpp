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

	// global and local range.
	sycl::nd_range ndr = sycl::nd_range{sycl::range{192, 14, 1}, sycl::range{192, 2, 1}};
	std::cout << "  == ndr=["
			  << ndr.get_global_range()[0] << ", " << ndr.get_global_range()[1] << ", " << ndr.get_global_range()[2]
			  << "], local_range=[" << ndr.get_local_range()[0] << ", " << ndr.get_local_range()[1] << ", "
			  << ndr.get_local_range()[2] << "]" << std::endl;

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
						cgh.parallel_for(ndr, k); });
}

int test_sycl_olc_interoperate_l0_backend_rope_ref()
{
	std::cout << "== Test: " << __FUNCTION__ << ":" << __LINE__ << std::endl;

	// It's hard to read for original cl file.
	// Convert to clean code via:
	// $ cpp original.cl > clean.cl
	std::string kernel_path = "../02_sycl_ocl_interoperate/src/kernel_rope_ref/";
	std::string kernel_source = load_kernel(kernel_path + "SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl");
	// std::cout << "  kernel_source = " << kernel_source << std::endl;

	auto queue = sycl::queue(sycl::gpu_selector_v);
	std::cout << "  == Using "
			  << queue.get_device().get_info<sycl::info::device::name>()
			  << ", Backend: " << queue.get_backend()
			  << std::endl;

	sycl::event ev;
	std::vector<std::pair<sycl::buffer<uint8_t, 1>, bool>> params;

	// gws=[1, 14, 192] lws=[1, 2, 192]
	// input0: data_type=i32, {1, 6, 1, 1, 1, 1, 14, 64, 0, 4, 1, 1, 1, 1, 1, 1, 6, 64, 1, 1, 1, 1, 1, 1, 6, 64, 1, 14, 1, 1, 1, 1, 6, 64}
	// input1: data_type=f16;format=bfyx;shape=[1,6,14,64];program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src0.txt
		// pad_l=[0, 0, 0, 0, 0, 0, 0, 0, 0, ];
		// pad_u=[0, 0, 4, 0, 0, 0, 0, 0, 0, ];
		// dyn_pad_dims=[000000100];
	// input2: data_type=f16;format=bfyx;shape=[1,1,6,64];
	// input3: data_type=f16;format=bfyx;shape=[1,1,6,64];
	// input4: data_type=f16;format=bfyx;shape=[1,14,6,64];

	DumpData input0;
	input0.data = std::vector<float>({1, 6, 1, 1, 1, 1, 14, 64, 0, 4, 1, 1, 1, 1, 1, 1, 6, 64, 1, 1, 1, 1, 1, 1, 6, 64, 1, 14, 1, 1, 1, 1, 6, 64});
	input0.format = "bfyx";
	input0.shape = {34};

	auto input1 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src0.txt");
	auto input2 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src1.txt");
	auto input3 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src2.txt");
	auto output_expected = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_dst0.txt");

	auto buf0 = input0.to_int(queue);
	auto buf1 = input1.to_half(queue);
	auto buf2 = input2.to_half(queue);
	auto buf3 = input3.to_half(queue);
	sycl::buffer param_0(reinterpret_cast<uint8_t *>(buf0), sycl::range{input0.data.size() * sizeof(int)});
	sycl::buffer param_1(reinterpret_cast<uint8_t *>(buf1), sycl::range{input1.data.size() * sizeof(sycl::half)});
	sycl::buffer param_2(reinterpret_cast<uint8_t *>(buf2), sycl::range{input2.data.size() * sizeof(sycl::half)});
	sycl::buffer param_3(reinterpret_cast<uint8_t *>(buf3), sycl::range{input3.data.size() * sizeof(sycl::half)});
	params.push_back({param_0, false});
	params.push_back({param_1, false});
	params.push_back({param_2, false});
	params.push_back({param_3, false});

	// for (size_t i = 0; i < input2.data.size(); i++) {
	// 	std::cout << buf2[i] << ", ";
	// }
	// std::cout << std::endl;

	auto output_buf = sycl::malloc_shared<sycl::half>(output_expected.data.size(), queue);
	sycl::buffer param_4(reinterpret_cast<uint8_t *>(output_buf), sycl::range{output_expected.data.size() * sizeof(sycl::half)});
	params.push_back({param_4, true});

	auto ret_ev = launchOpenCLKernelOnline(queue, kernel_source, "rope_ref_11982042700243959200_0_0__sa", params, ev);
	ret_ev.wait();

	// Print IN/OUT
	std::cout << "  == Compare input and output:" << std::endl;
	int diff_num = 0;
	for (size_t i = 0; i < output_expected.data.size(); i++)
	{
		if (fabs(output_expected.data[i] - output_buf[i]) > 0.00001f)
		{
			diff_num++;
			std::cout << "    output_expected[" << i << "] = " << output_expected.data[i] << " VS output_buf[" << i << "] = " << output_buf[i] << std::endl;
		}
		if (diff_num > 5)
		{
			break;
		}
	}

	std::cout << std::endl;
	std::cout << "  == Input and Ouput are " << (diff_num == 0 ? "Same." : "Not Same.") << std::endl;
	sycl::free(buf0, queue);
	sycl::free(buf1, queue);
	sycl::free(buf2, queue);
	sycl::free(buf3, queue);
	sycl::free(output_buf, queue);
	return 0;
}