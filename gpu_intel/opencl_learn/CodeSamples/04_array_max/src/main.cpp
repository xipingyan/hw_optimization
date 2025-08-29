// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>

#include "kernel_io.hpp"
#include "my_ocl.hpp"
#include "my_common.hpp"

// #define LWS 128
#define LWS 256
float run_kernel(cl::CommandQueue &queue, cl::Context &context, cl::Kernel kernel, std::vector<float> &array_data)
{
	float output = 0;

	size_t gws = array_data.size();
	size_t lws = LWS;
	size_t group_sz = gws / lws;
	// std::cout << "  == Run kernel:" << std::endl;
	std::cout << "  group_sz = " << group_sz << std::endl;
	std::cout << "  gws = " << gws << std::endl;
	std::cout << "  lws = " << lws << std::endl;

	// Create buffers on the device
	cl::Buffer buffer_IN_1(context, CL_MEM_READ_ONLY, sizeof(float) * array_data.size());
	cl::Buffer buffer_IN_2(context, CL_MEM_READ_WRITE, sizeof(float) * lws);
	cl::Buffer buffer_OUT(context, CL_MEM_READ_WRITE, sizeof(float));

	// Write to device
	queue.enqueueWriteBuffer(buffer_IN_1, CL_TRUE, 0, sizeof(float) * array_data.size(), array_data.data());
	queue.enqueueWriteBuffer(buffer_OUT, CL_TRUE, 0, sizeof(float), &output);

	kernel.setArg(0, buffer_IN_1);
	kernel.setArg(1, buffer_IN_2);
	kernel.setArg(2, buffer_OUT);
	kernel.setArg(3, array_data.size()); // input array size.

	// If lws is null, means = gws.
	// queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(array_data.size()), cl::NullRange);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(gws), cl::NDRange(lws));
	queue.finish();

	// Copy output from device to host
	queue.enqueueReadBuffer(buffer_OUT, CL_TRUE, 0, sizeof(float), &output);

	return output;
}

#include <algorithm>
float run_ref(std::vector<float> &array_input)
{
	std::cout << "  == Run calc max of array reference." << std::endl;
	auto t1 = std::chrono::high_resolution_clock::now();
	auto max_iter = std::max_element(array_input.begin(), array_input.end());
	if (max_iter != array_input.end())
	{
		auto t2 = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "  CPU host time = " << diff << " micr sec." << std::endl;
		return *max_iter;
	}
	return 0;
}

int main()
{
	std::cout << "== Test array max algorithm. " << std::endl;

	std::string kernel_fn = "../04_array_max/src/array_max_kernel.cl";
	std::string kernel_entry = "get_array_max";

	auto default_device = get_gpu_device();
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	std::cout << "== Create context" << std::endl;
	cl::Context context({default_device});

	std::cout << "== Create Sources" << std::endl;
	cl::Program::Sources sources;

	std::string kernel_code = load_kernel_source_codes(kernel_fn);
	if (kernel_code.empty())
	{
		std::cout << "== Fail: can't load: " << kernel_fn.c_str() << std::endl;
		exit(0);
	}

	std::cout << "== Put kernel string to source." << std::endl;
	sources.push_back({kernel_code.c_str(), kernel_code.length()});

	std::cout << "== Construct program with source and context." << std::endl;
	cl::Program program(context, sources);
	if (program.build({default_device}) != CL_SUCCESS)
	{
		std::cout << "  Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}

	// Construct kernel 1
	cl::vector<cl::Kernel> kernels;
	program.createKernels(&kernels);
	if (kernels.size() > 0)
	{
		auto kernel_name = kernels[0].getInfo<CL_KERNEL_FUNCTION_NAME>();
		std::cout << "  == Get kernel function name from  = " << kernel_name << std::endl;
	}

	std::cout << "== Create command queue" << std::endl;
	// create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	std::cout << "== Create Kernel with program and run." << std::endl;
	// alternative way to run the kernel
	cl::Kernel max_kernel = cl::Kernel(program, kernel_entry);

	auto kernel_name = max_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
	std::cout << "== Test get kernel name from cl::Kernel, kernel_name = " << kernel_name << std::endl;

	// ==================
	std::cout << "== Start to run." << std::endl;

	std::vector<float> array_input = {
        // Batch 0, 4x4 kernel matrix
        0.8f, 0.3f, 0.1f, 0.2f,  // token 0 row
        0.3f, 0.9f, 0.4f, 0.1f,  // token 1 row
        0.1f, 0.4f, 0.7f, 0.5f,  // token 2 row
        0.2f, 0.1f, 0.5f, 0.6f   // token 3 row
    };
	array_input = generate_vec(LWS * 30);

	auto max_ref = run_ref(array_input);
	auto max_ocl = run_kernel(queue, context, max_kernel, array_input);
	int64_t sum_tm = 0;
	int loop_num = 10;
	for (int i = 0; i < loop_num; i++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		max_ocl = run_kernel(queue, context, max_kernel, array_input);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "== [" << i << "] host time = " << diff << " micr sec." << std::endl;
		sum_tm += diff;
	}
	if (loop_num > 0)
		std::cout << "  Mean time = " << sum_tm / loop_num << " micr sec." << std::endl;

	std::cout << "== Done. " << std::endl;
	bool bclose = is_close<float>({max_ref}, {max_ocl});
	std::cout << "== Result Ref VS OCL: " << (bclose ? "success." : "fail.") << std::endl;
	if (!bclose)
		std::cout << "    max_ref = " << max_ref << ", max_ocl = " << max_ocl << std::endl;

	// A770 performance compare
	// get_array_max_1: host time: 1514 micr sec. kernel time: 526 micro sec.
	// get_array_max_2: host time: 939  micr sec. kernel time: 14  micro sec.
	return 0;
}