// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>

#include "kernel_io.hpp"
#include "dpp_ref.hpp"
#include "mat_diagonal_max_ref.hpp"
#include "my_log.hpp"

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

float run_max_mat_diagonal_kernel(cl::CommandQueue &queue, cl::Context &context, cl::Kernel kernel_max, Tensor &mat)
{
	#define LWS 128
	int gws = (mat.m + LWS - 1) / LWS * LWS;
	// create buffers on the device
	cl::Buffer buffer_mat(context, CL_MEM_READ_ONLY, sizeof(float) * mat.get_size());
	cl::Buffer buffer_local(context, CL_MEM_READ_WRITE, sizeof(float) * LWS);
	cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, sizeof(float) * 1);

	// write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, sizeof(float) * mat.get_size(), mat.data);

	kernel_max.setArg(0, buffer_mat);
	kernel_max.setArg(1, buffer_local);
	kernel_max.setArg(2, buffer_output);
	kernel_max.setArg(3, mat.m);

	// auto gws_nd = cl::NDRange(gws);
	// auto lws_nd = cl::NDRange(LWS);
	auto gws_nd = cl::NDRange(gws, 1, 1);
	auto lws_nd = cl::NDRange(LWS, 1, 1);
	print_nd_range(gws_nd);
	print_nd_range(lws_nd);

	for (auto i = 0; i < 10; i++) {
		auto t1 = std::chrono::high_resolution_clock::now();
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, gws_nd, lws_nd);
		queue.finish();
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "Run [" << i << "], tm = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
	}

	std::cout << "  == Start to copy output from device to host" << std::endl;
	float output = 0;
	// Copy output from device to host
	queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * 1, &output);
	return output;
}

std::vector<int> run_dpp_kernel(cl::CommandQueue &queue, cl::Context &context, cl::Kernel kernel_dpp, Tensor &mat, int selected_token_num = 0)
{
	assert(mat.m == mat.n);
	float numerical_threshold = 1e-6f;

	size_t total_tokens_num = mat.m;
	if (selected_token_num == 0) {
		selected_token_num = static_cast<size_t>(total_tokens_num * 0.6);
	}
	std::vector<int> output_ids(selected_token_num, -1);

	#define LWS_1 2
	int gws_1 = (mat.m + LWS_1 - 1) / LWS_1 * LWS_1;

	// create buffers on the device
	cl::Buffer buffer_mat(context, CL_MEM_READ_ONLY, sizeof(float) * mat.get_size());
	cl::Buffer buffer_cis(context, CL_MEM_READ_WRITE, sizeof(float) * selected_token_num * total_tokens_num);
	cl::Buffer buffer_di2s(context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens_num);  // diagonal value.
	cl::Buffer buffer_output_ids(context, CL_MEM_READ_WRITE, sizeof(int) * selected_token_num);
	cl::Buffer buffer_local_values(context, CL_MEM_READ_WRITE, sizeof(float) * LWS_1);
	cl::Buffer buffer_local_ids(context, CL_MEM_READ_WRITE, sizeof(int) * LWS_1);
	cl::Buffer buffer_best_value(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_best_id(context, CL_MEM_READ_WRITE, sizeof(int));

	// write mat to the device
	queue.enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, sizeof(float) * mat.get_size(), mat.data);
	
	kernel_dpp.setArg(0, buffer_mat);
	kernel_dpp.setArg(1, buffer_cis);
	kernel_dpp.setArg(2, buffer_di2s);
	kernel_dpp.setArg(3, buffer_output_ids);
	kernel_dpp.setArg(4, mat.b);
	kernel_dpp.setArg(5, mat.m);
	kernel_dpp.setArg(6, selected_token_num);
	kernel_dpp.setArg(7, buffer_local_values);
	kernel_dpp.setArg(8, buffer_local_ids);
	kernel_dpp.setArg(9, buffer_best_value);
	kernel_dpp.setArg(10, buffer_best_id);
	kernel_dpp.setArg(11, numerical_threshold);

	std::cout << "  == Params:" << std::endl;
	std::cout << "     gws_1 = " << gws_1 << std::endl;
	std::cout << "     LWS_1 = " << LWS_1 << std::endl;
	std::cout << "     selected_token_num = " << selected_token_num << std::endl;
	std::cout << "     M = " << mat.m << std::endl;

	queue.enqueueNDRangeKernel(kernel_dpp, cl::NullRange, cl::NDRange(mat.b, gws_1, 1), cl::NDRange(mat.b, LWS_1, 1));
	queue.finish();

	std::cout << "  == Start to copy output from device to host" << std::endl;
	// Copy output from device to host
	queue.enqueueReadBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());
	
	float best_value = 0;
	int best_id = -1;
	queue.enqueueReadBuffer(buffer_best_value, CL_TRUE, 0, sizeof(float), &best_value);
	queue.enqueueReadBuffer(buffer_best_id, CL_TRUE, 0, sizeof(int), &best_id);
	std::cout << "  == best_id = " << best_id << ", best_value = " << best_value << std::endl;

	return output_ids;
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

std::vector<int> run_ref(Tensor& mat, int selected_token_num = 0) {
	std::cout << "  == run ref" << std::endl;
	Config config;
	// Initialize config for testing
	config.visual_tokens_retain_percentage = 60;  // Will keep 3 out of 4 tokens
	config.relevance_weight = 0.5f;
	config.enable_pruning = true;
	config.pruning_debug_mode = true;
	config.use_negative_relevance = false;  // Not using negative correlation as requested
	config.numerical_threshold = 1e-6f;
	config.device = "CPU";
	config.use_ops_model = false;

	auto dpp_selector = std::make_unique<FastGreedyDPP>(config);

	if (selected_token_num == 0) {
		selected_token_num = config.visual_tokens_retain_percentage * mat.m;
	} 
	auto selected_tokens = dpp_selector->select(mat, selected_token_num);

	std::vector<int> concatenated_vec;
	for (auto st : selected_tokens) {
		for (auto s : st) {
			concatenated_vec.emplace_back(static_cast<int>(s));
		}
	}
	return concatenated_vec;
}

void print_result(std::vector<int> rslts, std::string prefix) {
	// std::cout << "== " << prefix.c_str() << std::endl;
	// for (size_t i = 0; i < rslts.size(); i++) {
	// 	std::cout << "  [" << i << "]: ";
	// 	for (size_t j = 0; j < std::min((size_t)8, rslts[i].size()); j++)
	// 	{
	// 		std::cout << rslts[i][j] << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }
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

	std::cout << "== Put mat string to source." << std::endl;
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
		std::cout << "== Get all kernels in kernel_fn" << std::endl;
		for (auto cl_kernel : kernels) {
			auto kernel_name = cl_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
			std::cout << "  == kernel name = " << kernel_name << std::endl;
		}
	}

	std::cout << "== Create command queue" << std::endl;
	// create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);


	std::string kernel_entry = "get_mat_diagonal_max_2";
	kernel_entry = "dpp_kernel";
	std::cout << "== Create Kernel with program and run." << std::endl;
	// alternative way to run the kernel
	cl::Kernel dpp_kernel = cl::Kernel(program, kernel_entry);

	auto kernel_name = dpp_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
	std::cout << "== Test get kernel name from cl::Kernel, kernel_name = " << kernel_name << std::endl;
	// auto old_kernel_program = dpp_kernel.getInfo<CL_KERNEL_PROGRAM>();

	// ==================
	std::cout << "== Start to run." << std::endl;

	int m = 1024;
	m= 4000;
	m = 8;
	auto mat = Tensor(1, m, m);
	mat.random_data();

	int selected_token_num = 1;
	auto selected_token_ref = run_ref(mat, selected_token_num);
	print_result(selected_token_ref, "selected_token_ref");

// mat diagonal max
#if 0
	{
		float ref_max = 0;
		CMatDiagMax(mat.m, mat.n, mat.data).get_max_val(ref_max);

		float gpu_max = run_max_mat_diagonal_kernel(queue, context, dpp_kernel, mat);
		std::cout << "== ref_max = " << ref_max << ", gpu_max = " << gpu_max << std::endl;
		std::cout << (ref_max == gpu_max ? "== Success" : "== Fail.") << std::endl;
		return 0;
	}
#else
	auto selected_token_gpu = run_dpp_kernel(queue, context, dpp_kernel, mat, selected_token_num);
	print_result(selected_token_gpu, "selected_token_gpu");
#endif

	std::cout << "== Done." << std::endl;
	return 0;
}