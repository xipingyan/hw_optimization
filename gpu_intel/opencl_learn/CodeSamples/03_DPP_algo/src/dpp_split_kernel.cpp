// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>

#include "kernel_io.hpp"
#include "dpp_ref.hpp"
#include "my_log.hpp"
#include "my_common.hpp"
#include "my_ocl.hpp"

inline cl::Kernel get_kernel_argmax(CMyTest &my_ocl)
{
	std::string kernel_entry = "get_array_max_single_group";
	return my_ocl.get_kernel(kernel_entry);
}

std::vector<int> run_dpp_split_kernel(Tensor &mat, int selected_token_num)
{
	std::string kernel_fn = "../03_DPP_algo/src/dpp_kernel_split.cl";
	std::string kernel_entry = "update_orthogonal_vector";
	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	float numerical_threshold = 1e-6f;

	size_t total_tokens_num = mat.m;
	if (selected_token_num == 0) {
		selected_token_num = static_cast<size_t>(total_tokens_num * 0.6);
	}
	std::vector<int> output_ids(selected_token_num, -1);

	auto kernel_argmax = get_kernel_argmax(my_ocl);
	auto kernel_update_orthogonal_vector = my_ocl.get_kernel();

	auto context = my_ocl.get_context();
	// ** prepare kernel argmax ***********************************
	std::vector<float> vec_di2s(total_tokens_num);
	for (int i = 0; i < total_tokens_num; i++) {
		vec_di2s[i] = mat.data[i * total_tokens_num + i];
	}
	cl::Buffer buffer_di2s(context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens_num);
	cl::Buffer buffer_best_value(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_best_id(context, CL_MEM_READ_WRITE, sizeof(int));
	cl::NDRange argmax_lws = cl::NDRange(1, 1024, 1);
	cl::NDRange argmax_gws = cl::NDRange(1, 1024, 1);
	kernel_argmax.setArg(0, buffer_di2s);
	kernel_argmax.setArg(1, sizeof(float) * 1024, nullptr);
	kernel_argmax.setArg(2, sizeof(int) * 1024, nullptr);
	kernel_argmax.setArg(3, buffer_best_value);
	kernel_argmax.setArg(4, buffer_best_id);
	kernel_argmax.setArg(5, total_tokens_num);

	// prepare kernel update orthogonal vector
	cl::Buffer buffer_mat(context, CL_MEM_READ_ONLY, sizeof(float) * mat.get_size());
	cl::Buffer buffer_cis(context, CL_MEM_READ_WRITE, sizeof(float) * selected_token_num * total_tokens_num);
	cl::Buffer buffer_output_ids(context, CL_MEM_READ_WRITE, sizeof(int) * selected_token_num);
	cl::NDRange lws = cl::NDRange(1, std::min(mat.m, 16), 1);
	cl::NDRange gws = cl::NDRange(mat._b, mat.m, 1);
	kernel_update_orthogonal_vector.setArg(0, buffer_mat);
	kernel_update_orthogonal_vector.setArg(1, mat.m);
	kernel_update_orthogonal_vector.setArg(2, buffer_best_id);
	kernel_update_orthogonal_vector.setArg(4, buffer_cis);
	kernel_update_orthogonal_vector.setArg(5, buffer_di2s);
	kernel_update_orthogonal_vector.setArg(6, numerical_threshold);

	// prepare update_marginal_gains
	cl::Kernel kernel_3 = my_ocl.get_kernel("update_marginal_gains");
	kernel_3.setArg(1, mat.m);
	kernel_3.setArg(2, buffer_best_id);
	kernel_3.setArg(3, buffer_cis);
	kernel_3.setArg(4, buffer_di2s);
	kernel_3.setArg(5, buffer_output_ids);

	for (int l = 0; l < 3; l++)
	{
		my_ocl.get_queue()->enqueueWriteBuffer(buffer_di2s, CL_TRUE, 0, sizeof(float) * total_tokens_num, vec_di2s.data());
		my_ocl.get_queue()->enqueueWriteBuffer(buffer_mat, CL_TRUE, 0, sizeof(float) * mat.get_size(), mat.data);
		my_ocl.get_queue()->enqueueWriteBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());

		auto t1 = std::chrono::high_resolution_clock::now();
		std::vector<cl::Event> eventList;
		for (size_t t = 0; t < selected_token_num; ++t)
		{
			// Step 1: argmax
			cl::Event eventA;
			my_ocl.get_queue()->enqueueNDRangeKernel(kernel_argmax, cl::NullRange, argmax_gws, argmax_lws, &eventList, &eventA);
			eventList.push_back(eventA);
			
			cl::Event eventB;
			// Step 2: update orthogonal vector
			kernel_update_orthogonal_vector.setArg(3, t);
			my_ocl.get_queue()->enqueueNDRangeKernel(kernel_update_orthogonal_vector, cl::NullRange, gws, lws, &eventList, &eventB);
			eventList.push_back(eventB);

			// Step 3:
			cl::Event eventC;
			kernel_3.setArg(0, t);
			my_ocl.get_queue()->enqueueNDRangeKernel(kernel_3, cl::NullRange, gws, lws, &eventList, &eventC);
			eventList.push_back(eventC);
		}
		my_ocl.get_queue()->finish();
		my_ocl.get_queue()->enqueueReadBuffer(buffer_output_ids, CL_TRUE, 0, sizeof(int) * selected_token_num, output_ids.data());

		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "  == tm = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
	}
	return output_ids;
}