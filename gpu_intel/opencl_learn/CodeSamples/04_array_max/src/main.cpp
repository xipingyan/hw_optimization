// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>

#include "kernel_io.hpp"
#include "my_ocl.hpp"
#include "my_common.hpp"

float run_ref(std::vector<float> &array_input)
{
	auto max_iter = std::max_element(array_input.begin(), array_input.end());
	if (max_iter != array_input.end())
	{
		return *max_iter;
	}
	return 0;
}

MyDevInfo g_dev_info = get_device_info();
#define ARRAY_SIZE 256 * 30

float run_kernel(CMyTest &my_ocl, std::vector<float> &array_data)
{
	auto kernel = my_ocl.get_kernel();
	auto preferred_lws = get_kernel_perferred_workgroup_size_multiple(kernel, my_ocl.get_device());
	std::vector<float> output;
	float out_max;

	size_t gws = array_data.size();
	size_t lws = std::min(array_data.size(), preferred_lws);
	size_t group_sz = gws / lws;
	output = std::vector<float>(group_sz, 0);

	// std::cout << "  == Run kernel:" << std::endl;
	std::cout << "  group_sz = " << group_sz << std::endl;
	std::cout << "  gws = " << gws << std::endl;
	std::cout << "  lws = " << lws << std::endl;
	std::cout << "  array_size = " << array_data.size() << std::endl;

	// Create buffers on the device
	cl::Buffer buffer_IN_1(my_ocl.get_context(), CL_MEM_READ_ONLY, sizeof(float) * array_data.size());
	cl::Buffer buffer_OUT(my_ocl.get_context(), CL_MEM_READ_WRITE, sizeof(float) * group_sz);

	// Write to device
	my_ocl.get_queue()->enqueueWriteBuffer(buffer_IN_1, CL_TRUE, 0, sizeof(float) * array_data.size(), array_data.data());
	my_ocl.get_queue()->enqueueWriteBuffer(buffer_OUT, CL_TRUE, 0, sizeof(float), output.data());

	kernel.setArg(0, buffer_IN_1);
	kernel.setArg(1, sizeof(float) * lws, nullptr);
	kernel.setArg(2, buffer_OUT);
	kernel.setArg(3, array_data.size()); // input array size.

	int64_t sum_tm = 0;
	int loop_num = 10;

	auto t2 = std::chrono::high_resolution_clock::now();
	auto run_ocl = [&]()
	{
		// If lws is null, means = gws.
		// queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(array_data.size()), cl::NullRange);
		my_ocl.get_queue()->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws));
		my_ocl.get_queue()->finish();

		// Copy output from device to host
		my_ocl.get_queue()->enqueueReadBuffer(buffer_OUT, CL_TRUE, 0, sizeof(float) * group_sz, output.data());

		t2 = std::chrono::high_resolution_clock::now();
		// 跨EU没有办法同步，所以每个group的结果，都要返回，在cpu侧，再重新计算一次。
		out_max = run_ref(output);
	};

	// warmup
	run_ocl();

	for (int i = 0; i < loop_num; i++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();

		run_ocl();

		auto t3 = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count();
		auto tm_gpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		auto tm_ref = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
		std::cout << "  == [" << i << "] host time = " << diff << " micr sec." << ", tm_gpu: " << tm_gpu << ", tm_ref: " << tm_ref << std::endl;
		sum_tm += diff;
	}
	if (loop_num > 0)
		std::cout << "  Mean time = " << sum_tm / loop_num << " micr sec." << std::endl;

	return out_max;
}

// Test: 跨group求max，适用于单个kernel，在host端同步。
void test_array_max_all_group()
{
	std::string kernel_fn = "../04_array_max/src/array_max_kernel.cl";
	std::string kernel_entry = "get_array_max";

	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	// ==================
	std::vector<float> array_input = {
		// Batch 0, 4x4 kernel matrix
		0.8f, 0.3f, 0.1f, 0.2f, // token 0 row
		0.3f, 0.9f, 0.4f, 0.1f, // token 1 row
		0.1f, 0.4f, 0.7f, 0.5f, // token 2 row
		0.2f, 0.1f, 0.5f, 0.6f	// token 3 row
	};
	array_input = generate_vec(ARRAY_SIZE);

	std::cout << "== Start to run Reference." << std::endl;
	float max_ref = 0;
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		max_ref = run_ref(array_input);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "  CPU host time = " << diff << " micr sec." << std::endl;
	}

	std::cout << "== Start to run run_kernel" << std::endl;
	auto max_ocl = run_kernel(my_ocl, array_input);

	std::cout << "== Done. " << std::endl;
	bool bclose = is_close<float>({max_ref}, {max_ocl});
	std::cout << "== Result Ref VS OCL: " << (bclose ? "success." : "fail.") << std::endl;
	if (!bclose)
		std::cout << "    max_ref = " << max_ref << ", max_ocl = " << max_ocl << std::endl;

	// GPU: A770 performance compare
	// get_array_max_1: host time: 1514 micr sec. kernel time: 526 micro sec.
	// get_array_max_2: host time: 939  micr sec. kernel time: 14  micro sec.

	// Kernel: get_array_max_2
	// array size = 7680
	// group_sz = 30
	// gws = 7680
	// lws = 256
	// Mean time = 610 micr sec.
}

// Test: 单个group求max，复杂的kernel，需要再kernel内部求最大值，kernel内部同步，kernel内的后续步骤需要这个最大值。
void test_array_max_single_group()
{
	std::string kernel_fn = "../04_array_max/src/array_max_single_group_kernel.cl";
	std::string kernel_entry = "get_array_max_single_group";

	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	// ==================
	int arr_size = ARRAY_SIZE;
	// arr_size = 9;
	auto array_input = generate_vec(arr_size);
	auto max_ref = run_ref(array_input);
	std::cout << "== max_ref = " << max_ref << std::endl;

	auto run_kernel = [&]()
	{
		auto kernel = my_ocl.get_kernel();
		float output_max_val = -INFINITY;
		float output_max_id = -1;

		size_t gws = g_dev_info.device_max_group_size;
		size_t lws = g_dev_info.device_max_group_size;
		// gws = lws = 4;
		size_t group_sz = gws / lws;
		// std::cout << "  == Run kernel:" << std::endl;
		std::cout << "  group_sz = " << group_sz << std::endl;
		std::cout << "  gws = " << gws << std::endl;
		std::cout << "  lws = " << lws << std::endl;
		std::cout << "  arr_size = " << arr_size << std::endl;

		// Create buffers on the device
		cl::Buffer buffer_arr(my_ocl.get_context(), CL_MEM_READ_ONLY, sizeof(float) * array_input.size());
		cl::Buffer buffer_out_max_val(my_ocl.get_context(), CL_MEM_READ_WRITE, sizeof(float));
		cl::Buffer buffer_out_max_id(my_ocl.get_context(), CL_MEM_READ_WRITE, sizeof(int));

		// Write to device
		my_ocl.get_queue()->enqueueWriteBuffer(buffer_arr, CL_TRUE, 0, sizeof(float) * array_input.size(), array_input.data());

		kernel.setArg(0, buffer_arr);
		kernel.setArg(1, sizeof(float) * lws, nullptr); // __local 只要指定大小即可，不需要分配具体的内存，如果分配了，反而能引起kernel执行错误。
		kernel.setArg(2, sizeof(int) * lws, nullptr);
		kernel.setArg(3, buffer_out_max_val);
		kernel.setArg(4, buffer_out_max_id);
		kernel.setArg(5, static_cast<int>(array_input.size())); // input array size.

		int loop_num = 10;
		int64_t sum_tm = 0;
		for (int i = 0; i < loop_num; i++)
		{
			auto t1 = std::chrono::high_resolution_clock::now();
			my_ocl.get_queue()->enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws));
			my_ocl.get_queue()->finish();
			auto t2 = std::chrono::high_resolution_clock::now();
			auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			std::cout << "== [" << i << "] host time = " << diff << " micr sec." << std::endl;
			sum_tm += diff;
		}
		if (loop_num > 0)
			std::cout << "  Mean time = " << sum_tm / loop_num << " micr sec." << std::endl;

		my_ocl.get_queue()->enqueueReadBuffer(buffer_out_max_val, CL_TRUE, 0, sizeof(float), &output_max_val);
		my_ocl.get_queue()->enqueueReadBuffer(buffer_out_max_id, CL_TRUE, 0, sizeof(int), &output_max_id);
		return output_max_val;
	};

	auto max_ocl = run_kernel();

	std::cout << "== Done. " << std::endl;
	bool bclose = is_close<float>({max_ref}, {max_ocl});
	std::cout << "== Result Ref VS OCL: " << (bclose ? "success." : "fail.") << std::endl;
	if (!bclose)
		std::cout << "    max_ref = " << max_ref << ", max_ocl = " << max_ocl << std::endl;

	// A770 performance compare
	// get_array_max_1: host time: 1514 micr sec. kernel time: 526 micro sec.
	// get_array_max_2: host time: 939  micr sec. kernel time: 14  micro sec.
}

int main()
{
	std::cout << "== Test array max algorithm. " << std::endl;
	std::cout << "  Macro:" << std::endl;
	std::cout << "    SINGLE_GROUP:1  means 1 EU process all array elements. because only one EU can sync with barrier" << std::endl;

	bool enable_singel_group = false;
	get_env_bool("SINGLE_GROUP", enable_singel_group);

	if (enable_singel_group)
	{
		std::cout << "== Run: test_array_max_single_group" << std::endl;
		test_array_max_single_group();
	}
	else
	{
		std::cout << "== Run: test_array_max_all_group" << std::endl;
		test_array_max_all_group(); // 如果数据> max group size(1024), 结果就有可能是错误的。所以规约算法只能在一个group内完成。
	}

	return 0;
}