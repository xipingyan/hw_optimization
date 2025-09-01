// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>

#include "kernel_io.hpp"
#include "dpp_ref.hpp"
#include "mat_diagonal_max_ref.hpp"
#include "my_log.hpp"
#include "my_common.hpp"
#include "my_ocl.hpp"

static size_t g_max_ws_in_one_group[3] = {0};

std::vector<int> run_dpp_kernel(CMyTest& my_olc, Tensor &mat, int selected_token_num = 0)
{
	assert(mat.m == mat.n);
	float numerical_threshold = 1e-6f;

	size_t total_tokens_num = mat.m;
	if (selected_token_num == 0) {
		selected_token_num = static_cast<size_t>(total_tokens_num * 0.6);
	}
	std::vector<int> output_ids(selected_token_num, -1);

	int lws_1 = std::min(mat.m, (int)g_max_ws_in_one_group[1]);
	int gws_1 = lws_1;
	// lws_1 = 4;
	// gws_1 = 4;

	// create buffers on the device
	auto context = my_olc.get_context();
	cl::Buffer buffer_mat(context, CL_MEM_READ_ONLY, sizeof(float) * mat.get_size());
	cl::Buffer buffer_cis(context, CL_MEM_READ_WRITE, sizeof(float) * selected_token_num * total_tokens_num);
	cl::Buffer buffer_di2s(context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens_num);  // diagonal value.
	cl::Buffer buffer_output_ids(context, CL_MEM_READ_WRITE, sizeof(int) * selected_token_num);
	cl::Buffer buffer_best_value(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_best_id(context, CL_MEM_READ_WRITE, sizeof(int));

	// write mat to the device
	my_olc.get_queue()->enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, sizeof(float) * mat.get_size(), mat.data);
	my_olc.get_queue()->enqueueWriteBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());
	
	auto kernel_dpp = my_olc.get_kernel();
	kernel_dpp.setArg(0, buffer_mat);
	kernel_dpp.setArg(1, buffer_cis);
	kernel_dpp.setArg(2, buffer_di2s);
	kernel_dpp.setArg(3, buffer_output_ids);
	kernel_dpp.setArg(4, mat.b);
	kernel_dpp.setArg(5, mat.m);
	kernel_dpp.setArg(6, selected_token_num);
	kernel_dpp.setArg(7, sizeof(float) * lws_1, nullptr);
	kernel_dpp.setArg(8, sizeof(int) * lws_1, nullptr);
	kernel_dpp.setArg(9, buffer_best_value);
	kernel_dpp.setArg(10, buffer_best_id);
	kernel_dpp.setArg(11, numerical_threshold);

	std::cout << "  == Params:" << std::endl;
	std::cout << "     gws_1 = " << gws_1 << std::endl;
	std::cout << "     lws_1 = " << lws_1 << std::endl;
	std::cout << "     selected_token_num = " << selected_token_num << std::endl;
	std::cout << "     M = " << mat.m << std::endl;

	for (int l = 0; l < 1; l++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		my_olc.get_queue()->enqueueNDRangeKernel(kernel_dpp, cl::NullRange, cl::NDRange(mat.b, gws_1, 1), cl::NDRange(mat.b, lws_1, 1));
		my_olc.get_queue()->finish();
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "  == tm = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	}

	std::cout << "  == Start to copy output from device to host" << std::endl;
	// Copy output from device to host
	my_olc.get_queue()->enqueueReadBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());
	std::cout << "      Copy from device to host finish. size = " << output_ids.size() << std::endl;
	
	// std::sort(output_ids.begin(), output_ids.end());

	// for (auto outp_id : output_ids) {
	// 	std::cout << "  == output_ids = " << outp_id << std::endl;
	// }

	return output_ids;
}


std::vector<int> run_ref(Tensor& mat, int selected_token_num = 0) {
	Config config;
	// Initialize config for testing
	config.visual_tokens_retain_percentage = 60;  // Will keep 3 out of 4 tokens
	config.relevance_weight = 0.5f;
	config.enable_pruning = true;
	config.pruning_debug_mode = false;
	config.use_negative_relevance = false;  // Not using negative correlation as requested
	config.numerical_threshold = 1e-6f;
	config.device = "CPU";
	config.use_ops_model = false;

	auto dpp_selector = std::make_unique<FastGreedyDPP>(config);

	if (selected_token_num == 0) {
		selected_token_num = config.visual_tokens_retain_percentage * mat.m;
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	auto selected_tokens = dpp_selector->select(mat, selected_token_num);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "  == CPU refer time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;

	std::vector<int> concatenated_vec;
	for (auto st : selected_tokens) {
		for (auto s : st) {
			concatenated_vec.emplace_back(static_cast<int>(s));
		}
	}
	return concatenated_vec;
}

int main()
{
	std::cout << "== Test DPP algorithm. " << std::endl;
	get_device_info(g_max_ws_in_one_group);

	std::string kernel_fn = "../03_DPP_algo/src/dpp_kernel.cl";
	std::string kernel_entry = "dpp_kernel";
	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	// ==================
	std::cout << "== Generate random test data." << std::endl;
	int m = 1024;
	// m = 40;
	m= 4000;
	// m = 9;
	auto mat = Tensor(1, m, m);
	mat.random_data();
	int selected_token_num = m * 0.6;
	// selected_token_num = 2;
	
	std::cout << "== Start to run DPP Reference." << std::endl;
	std::vector<int> selected_token_ref;
	selected_token_ref = run_ref(mat, selected_token_num);

	std::cout << "== Start to run DPP GPU kernel." << std::endl;
	auto selected_token_gpu = run_dpp_kernel(my_ocl, mat, selected_token_num);

	std::cout << "== Ref VS GPU result compare:" << std::endl;
	if (!is_same<int>(selected_token_ref, selected_token_gpu))
	{
		std::cout << "  == Fail, diff as follow:" << std::endl;
		print_diff<int>(selected_token_ref, selected_token_gpu);

		std::cout << "== Failed." << std::endl;
	}
	else
	{
		std::cout << "  == Success." << std::endl;
		std::cout << "== Done." << std::endl;
	}

	return 0;
}