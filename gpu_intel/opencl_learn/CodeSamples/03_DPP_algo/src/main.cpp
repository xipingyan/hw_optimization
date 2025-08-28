// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>

#include "kernel_io.hpp"
#include "dpp_ref.hpp"
#include "mat_diagonal_max_ref.hpp"

cl::Device get_gpu_device() {
	// get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0)
	{
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	size_t selected_platform = -1;
	for (size_t i = 0; i < all_platforms.size(); i++)
	{
		std::string platname = all_platforms[i].getInfo<CL_PLATFORM_NAME>();
		if (platname.find("Graphics") != std::string::npos)
		{
			selected_platform = i;
			break;
		}
	}
	if (selected_platform == -1)
	{
		std::cout << " No GPU platforms is found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[selected_platform];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
	if (all_devices.size() == 0)
	{
		std::cout << " No GPU device is found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];

	return default_device;
}

float run_max_mat_diagonal_kernel(cl::CommandQueue &queue, cl::Context &context, cl::Kernel kernel_max, Tensor &kernel)
{
	#define LWS 128
	int gws = (kernel.m + LWS - 1) / LWS * LWS;
	// create buffers on the device
	cl::Buffer buffer_mat(context, CL_MEM_READ_ONLY, sizeof(float) * kernel.get_size());
	cl::Buffer buffer_local(context, CL_MEM_READ_WRITE, sizeof(float) * LWS);
	cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, sizeof(float) * 1);

	// write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, sizeof(float) * kernel.get_size(), kernel.data);

	kernel_max.setArg(0, buffer_mat);
	kernel_max.setArg(1, buffer_local);
	kernel_max.setArg(2, buffer_output);
	kernel_max.setArg(3, kernel.m);

	queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(gws), cl::NDRange(LWS));
	queue.finish();

	// dump_kernel_bin(program);

	std::cout << "  == Start to copy output from device to host" << std::endl;
	float output = 0;
	// Copy output from device to host
	queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * 1, &output);
	return output;
}

std::vector<std::vector<size_t>> run_kernel(cl::CommandQueue &queue, cl::Context &context, cl::Kernel kernel_add, Tensor &kernel)
{
	// // create buffers on the device
	// cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * num);
	// cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * num);
	// cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * num);

	// // write arrays A and B to the device
	// queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * num, inputs[0].data());
	// queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * num, inputs[1].data());

	// kernel_add.setArg(0, buffer_A);
	// kernel_add.setArg(1, buffer_B);
	// kernel_add.setArg(2, buffer_C);

	// queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(num), cl::NullRange);
	// queue.finish();

	// // dump_kernel_bin(program);

	// std::cout << "  == Start to copy output from device to host" << std::endl;
	// // Copy output from device to host
	// queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * num, output.data());
	return {};
}

template <typename T>
bool is_close(const std::vector<T> &vec1, const std::vector<T> &vec2)
{
	// 1. Check if the sizes are different
	if (vec1.size() != vec2.size())
	{
		return false;
	}

	// 2. Iterate through elements and compare them
	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (vec1[i] != vec2[i])
		{
			return false; // Found a differing element
		}
	}

	// If we reach here, sizes are the same and all elements are equal
	return true;
}

std::vector<std::vector<size_t>> run_ref(Tensor& kernel) {
	std::cout << "  == run ref" << std::endl;
	Config config;
	// Initialize config for testing
	config.visual_tokens_retain_percentage = 75;  // Will keep 3 out of 4 tokens
	config.relevance_weight = 0.5f;
	config.enable_pruning = true;
	config.pruning_debug_mode = true;
	config.use_negative_relevance = false;  // Not using negative correlation as requested
	config.numerical_threshold = 1e-6f;
	config.device = "CPU";
	config.use_ops_model = false;

	auto dpp_selector = std::make_unique<FastGreedyDPP>(config);

	int num_tokens_to_keep = 3;
	auto selected_tokens = dpp_selector->select(kernel, num_tokens_to_keep);

	return selected_tokens;
}

void print_result(std::vector<std::vector<size_t>> rslts, std::string prefix) {
	std::cout << "== " << prefix.c_str() << std::endl;
	for (size_t i = 0; i < rslts.size(); i++) {
		std::cout << "  [" << i << "]: ";
		for (size_t j = 0; j < std::min((size_t)8, rslts[i].size()); j++)
		{
			std::cout << rslts[i][j] << ", ";
		}
		std::cout << std::endl;
	}
}

int main()
{
	std::cout << "== Test DPP algorithm. " << std::endl;

	auto default_device = get_gpu_device();
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	std::cout << "== Create context" << std::endl;
	cl::Context context({default_device});

	std::cout << "== Create Sources" << std::endl;
	cl::Program::Sources sources;

	std::string kernel_fn = "../03_DPP_algo/src/dpp_kernel.cl";
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


	std::string kernel_entry = "get_mat_diagonal_max_2";
	std::cout << "== Create Kernel with program and run." << std::endl;
	// alternative way to run the kernel
	cl::Kernel dpp_kernel = cl::Kernel(program, kernel_entry);

	auto kernel_name = dpp_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
	std::cout << "== Test get kernel name from cl::Kernel, kernel_name = " << kernel_name << std::endl;
	// auto old_kernel_program = dpp_kernel.getInfo<CL_KERNEL_PROGRAM>();

	// ==================
	std::cout << "== Start to run." << std::endl;

	auto kernel = Tensor(1, 1024, 1024);
	kernel.random_data();

	// mat diagonal max
	{
		float ref_max = 0;
		CMatDiagMax(1024, 1024, kernel.data).get_max_val(ref_max);

		float gpu_max = run_max_mat_diagonal_kernel(queue, context, dpp_kernel, kernel);
		std::cout << "== ref_max = " << ref_max << ", gpu_max = " << gpu_max << std::endl;
		std::cout << (ref_max == gpu_max ? "== Success" : "== Fail.") << std::endl;
		return 0;
	}

	auto selected_token_ref = run_ref(kernel);
	print_result(selected_token_ref, "selected_token_ref");

	auto selected_token_gpu = run_kernel(queue, context, dpp_kernel, kernel);
	print_result(selected_token_gpu, "selected_token_gpu");

	std::cout << "== Done." << std::endl;
	return 0;
}