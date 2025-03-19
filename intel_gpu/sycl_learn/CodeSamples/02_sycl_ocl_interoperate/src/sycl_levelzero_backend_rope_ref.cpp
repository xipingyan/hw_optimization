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

static bool get_env(std::string env) {
	auto env_str = std::getenv(env.c_str());
	if (env_str && std::string("1") == env_str) {
		std::cout << "  == Get: " << env << " = 1" << std::endl;
		return true;
	}
	std::cout << "  == Get: " << env << " = 0" << std::endl;
	return false;
}

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
	else
	{
		std::cout << "  == Fail: can't read kernle source: " << kernel_fn << std::endl;
		exit(0);
	}

	return std::string();
}

static sycl::event launchOpenCLKernelOnline(sycl::queue &q, std::string source,
											std::string func_name, std::vector<void*> &params,
											sycl::event &dep_event, bool test_performance)
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
	sycl::event ret_ev;
	size_t loop_num = test_performance ? 150 : 1;
	for (size_t idx = 0; idx < loop_num; idx++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		ret_ev = q.submit([&](sycl::handler &cgh)
				 {
                        cgh.depends_on(dep_event);

						for (int i = 0; i < params.size(); i++)
						{
							cgh.set_arg(i, params[i]);
						}

						// Invoke the kernel over an nd-range.
						cgh.parallel_for(ndr, k); });
		ret_ev.wait();
		auto t2 = std::chrono::high_resolution_clock::now();
		if (test_performance) {
			auto tm = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			if ( idx >= 100) {
				static int64_t sum = 0;
				static int count = 0;
				sum+= tm;
				count++;
				std::cout << "  == Infer " << idx << ", time = " << tm << " micro sec. mean: " << (float)sum / count << std::endl;
			}
			else
				std::cout << "  == Infer " << idx << ", time = " << tm << " micro sec." << std::endl;
		}
	}
	return ret_ev;
}

static sycl::event launchSyclKernel(sycl::queue &q, int *buf0, sycl::half *buf1, sycl::half *buf2, sycl::half *buf3, sycl::half *out_buf, bool test_performance)
{
	sycl::nd_range ndr = sycl::nd_range{sycl::range{1, 14, 192}, sycl::range{1, 2, 192}};
	auto* shape_info = buf0;
	auto* input = buf1;
	auto* cos = buf2;
	auto* sin = buf3;
	auto* output = out_buf;

	sycl::event ret_ev;
	size_t loop_num = test_performance ? 150 : 1;
	for (size_t i = 0; i < loop_num; i++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		q.submit([&](sycl::handler &cgh)
				 { 
					// auto out = sycl::stream(1024, 768, cgh);
					cgh.parallel_for<class rope_sycl>(ndr, [=](sycl::nd_item<3> itm)
													 {
										const uint b = itm.get_global_id(0);
										const uint h = itm.get_global_id(1);
										const uint p = (uint)itm.get_global_id(2) / 32;
										const uint r = (uint)itm.get_global_id(2) % 32;

										uint input_idx = ((1 * 0) + (64 * (shape_info[8])) + ((64 * (14 + (shape_info[8] + shape_info[9]))) * 0) + ((64 * (14 + (shape_info[8] + shape_info[9])) * 1) * 0) + ((64 * (14 + (shape_info[8] + shape_info[9])) * 1 * 1 * 1 * 1) * 0) + ((64 * (14 + (shape_info[8] + shape_info[9])) * 1 * 1 * 1 * 1 * (shape_info[1] + 0)) * 0)) + (0) * 1 + (h) * 64 + (p) * (64 * (14 + (shape_info[8] + shape_info[9])) * 1 * 1 * 1 * 1) + (b) * (64 * (14 + (shape_info[8] + shape_info[9])) * 1 * 1 * 1 * 1 * (shape_info[1] + 0));

										uint cos_sin_b = b < (shape_info[10]) ? b : 0;
										uint cos_sin_h = h < 1 ? h : 0;
										uint cos_sin_p = p;
										cos_sin_p = cos_sin_p < (shape_info[16]) ? cos_sin_p : 0;

										uint cos_sin_idx = ((1 * 0) + (64 * 0) + ((64 * (shape_info[16] + 0)) * 0) + ((64 * (shape_info[16] + 0) * 1) * 0) + ((64 * (shape_info[16] + 0) * 1 * 1 * 1 * 1) * 0) + ((64 * (shape_info[16] + 0) * 1 * 1 * 1 * 1 * 1) * 0)) + (0) * 1 + (cos_sin_p) * 64 + (cos_sin_h) * (64 * (shape_info[16] + 0) * 1 * 1 * 1 * 1) + (cos_sin_b) * (64 * (shape_info[16] + 0) * 1 * 1 * 1 * 1 * 1);
										uint cos_idx = cos_sin_idx;
										uint sin_idx = cos_sin_idx;

										uint output_idx = ((1 * 0) + (64 * 0) + ((64 * (shape_info[32] + 0)) * 0) + ((64 * (shape_info[32] + 0) * 1) * 0) + ((64 * (shape_info[32] + 0) * 1 * 1 * 1 * 1) * 0) + ((64 * (shape_info[32] + 0) * 1 * 1 * 1 * 1 * 14) * 0)) + (0) * 1 + (p) * 64 + (h) * (64 * (shape_info[32] + 0) * 1 * 1 * 1 * 1) + (b) * (64 * (shape_info[32] + 0) * 1 * 1 * 1 * 1 * 14);
										sycl::half in1 = input[input_idx + r];
										sycl::half in2 = input[input_idx + 32 + r];
										output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;
										output[output_idx + 32 + r] = cos[cos_idx + 32 + r] * in2 +
																	  sin[sin_idx + 32 + r] * in1;
										// out << "output[" << output_idx << "]=" << output[output_idx + r] << sycl::endl;
																	   }); })
			.wait();
		auto t2 = std::chrono::high_resolution_clock::now();
		if (test_performance)
			std::cout << "  == Infer " << i << ", time = " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro sec." << std::endl;
	}
	return ret_ev;
}

static sycl::event launchSyclKernel_exp_shape(sycl::queue &q, sycl::half *buf1, sycl::half *buf2, sycl::half *buf3, sycl::half *out_buf, bool test_performance)
{
	sycl::nd_range ndr = sycl::nd_range{sycl::range{1, 14, 192}, sycl::range{1, 2, 192}};
	auto* input = buf1;
	auto* cos = buf2;
	auto* sin = buf3;
	auto* output = out_buf;

	// 0, 1, 2, 3, 4, 5, 6,  7,  8, 9,10,11,12,13,14,15,16, 17, 18,19,20,21,22,23,24,25,26, 27,28,29,30,31,32, 33
	//{1, 6, 1, 1, 1, 1, 14, 64, 0, 4, 1, 1, 1, 1, 1, 1, 6, 64, 1, 1, 1, 1, 1, 1, 6, 64, 1, 14, 1, 1, 1, 1, 6, 64}

	sycl::event ret_ev;
	size_t loop_num = test_performance ? 15 : 1;
	for (size_t i = 0; i < loop_num; i++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		q.submit([&](sycl::handler &cgh)
				 { 
					// auto out = sycl::stream(1024, 768, cgh);
					cgh.parallel_for<class rope_sycl_exp_shape>(ndr, [=](sycl::nd_item<3> itm)
													 {
										const uint b = itm.get_global_id(0);
										const uint h = itm.get_global_id(1);
										const uint p = (uint)itm.get_global_id(2) / 32;
										const uint r = (uint)itm.get_global_id(2) % 32;

										uint input_idx = (h) * 64 + (p) * 1152 + (b) * 6912;

										uint cos_sin_b = b < 1 ? b : 0;
										uint cos_sin_h = h < 1 ? h : 0;
										uint cos_sin_p = p;
										cos_sin_p = cos_sin_p < 6 ? cos_sin_p : 0;

										uint cos_sin_idx = (cos_sin_p) * 64 + (cos_sin_h) * 384 + (cos_sin_b) * 384;
										uint cos_idx = cos_sin_idx;
										uint sin_idx = cos_sin_idx;

										uint output_idx = (p) * 64 + (h) * 384 + (b) * 5376;
										sycl::half in1 = input[input_idx + r];
										sycl::half in2 = input[input_idx + 32 + r];
										output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;
										output[output_idx + 32 + r] = cos[cos_idx + 32 + r] * in2 +
																	  sin[sin_idx + 32 + r] * in1;
										// out << "output[" << output_idx << "]=" << output[output_idx + r] << sycl::endl;
																	   }); })
			.wait();
		auto t2 = std::chrono::high_resolution_clock::now();
		if (test_performance)
			std::cout << "  == Infer " << i << ", time = " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro sec." << std::endl;
	}
	return ret_ev;
}

int test_sycl_olc_interoperate_l0_backend_rope_ref()
{
	std::cout << "  OCL_KERNEL=1:  [Default] SYCL pipeline + OpenCL C kernel." << std::endl;
	std::cout << "  SYCL_KERNEL=1: Test SYCL kernel in sycl pipeline.\n" << std::endl;
	std::cout << "== Test: " << __FUNCTION__ << ", " << __FILE__ << ":" << __LINE__ << std::endl;

	bool test_performance = get_env("PERFORMANCE");
	bool test_sycl_kernel = get_env("SYCL_KERNEL");
	if (get_env("OCL_KERNEL")) {
		test_sycl_kernel = false;
	}
	std::cout << "  == test_sycl_kernel = " << test_sycl_kernel << std::endl;

	// It's hard to read for original cl file.
	// Convert to clean code via:
	// $ cpp original.cl > clean.cl
	std::string kernel_path = "../src/kernel_rope_ref/";
	if (!check_path_exist(kernel_path))
	{
		kernel_path = "../02_sycl_ocl_interoperate/src/kernel_rope_ref/";
	}
	std::string kernel_source = load_kernel(kernel_path + "SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl");
	// std::cout << "  kernel_source = " << kernel_source << std::endl;

	auto queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
	std::cout << "  == Using "
			  << queue.get_device().get_info<sycl::info::device::name>()
			  << ", Backend: " << queue.get_backend()
			  << std::endl;

	std::cout << "  == queue in_order = " << queue.is_in_order()
			  << std::endl;

	sycl::event ev;
	std::vector<void*> params;

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

	auto input1 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_src0.txt");
	auto input2 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_src1.txt");
	auto input3 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_src2.txt");
	auto output_expected = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_dst0.txt");

	auto buf0 = input0.to_int(queue);
	auto buf1 = input1.to_half(queue);
	auto buf2 = input2.to_half(queue);
	auto buf3 = input3.to_half(queue);
	// for (size_t i = 0; i < input2.data.size(); i++) {
	// 	std::cout << buf2[i] << ", ";
	// }
	// std::cout << std::endl;
	auto* output_buf = sycl::malloc_shared<sycl::half>(output_expected.data.size(), queue);

	if (test_sycl_kernel)
	{
		std::cout << "  == run sycl kernel." << std::endl;
		// auto ret_ev = launchSyclKernel(queue, buf0, buf1, buf2, buf3, output_buf, test_performance);
		auto ret_ev = launchSyclKernel_exp_shape(queue, buf1, buf2, buf3, output_buf, test_performance);
		ret_ev.wait();
	}
	else
	{
		std::cout << "  == run OpenCL kernel." << std::endl;
		params.emplace_back(buf0);
		params.emplace_back(buf1);
		params.emplace_back(buf2);
		params.emplace_back(buf3);
		params.emplace_back(output_buf);

		auto ret_ev = launchOpenCLKernelOnline(queue, kernel_source, "rope_ref_11982042700243959200_0_0__sa", params, ev, test_performance);
		ret_ev.wait();

		// Still not finish. so just backup.
		// auto ret_ev = launchOpenCLKernelOnlineLevelZero(queue, kernel_source, "rope_ref_11982042700243959200_0_0__sa", params, ev, test_performance);
		// ret_ev.wait();
	}

	// Print IN/OUT
	if (!test_performance)
	{
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
	}

	sycl::free(buf0, queue);
	sycl::free(buf1, queue);
	sycl::free(buf2, queue);
	sycl::free(buf3, queue);
	sycl::free(output_buf, queue);
	return 0;
}