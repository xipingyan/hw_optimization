// Reference:

#include <stdio.h>
#include <iostream>

#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>

#include "kernel_io.hpp"
#include "my_log.hpp"
#include "my_common.hpp"
#include "my_ocl.hpp"
#include "gemm_opt_gpu.hpp"

static MyDevInfo g_dev_inf;

static bool g_enable_weight_trans = true;

// float/fp16
template<typename T>
static std::vector<T> run_gemm_kernel(CMyTest& my_olc, CGEMM_Ref::Ptr gemm_ref_ptr, std::string kernel_entry)
{
	auto M = gemm_ref_ptr->get_m();
	auto N = gemm_ref_ptr->get_n();
	auto K = gemm_ref_ptr->get_k();

	std::vector<T> output(M * N, 0);

	// Default reference.
	auto lws = cl::NDRange(1, 16, 1);
	auto gws = cl::NDRange(M, N, 1);
#define LWS_SZ 16
	lws = cl::NDRange(LWS_SZ, LWS_SZ, 1);
	gws = cl::NDRange((N + LWS_SZ - 1) / LWS_SZ * LWS_SZ, std::max(LWS_SZ, (M + LWS_SZ - 1) / LWS_SZ * LWS_SZ));

	// create buffers on the device
	auto context = my_olc.get_context();
	cl::Buffer buffer_intput(context, CL_MEM_READ_ONLY, sizeof(T) * M * K);
	cl::Buffer buffer_weight(context, CL_MEM_READ_ONLY, sizeof(T) * K * N);
	cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, sizeof(T) * M * N);

	// write mat to the device
	my_olc.get_queue()->enqueueWriteBuffer(buffer_intput, CL_TRUE, 0, sizeof(T) * M * K, gemm_ref_ptr->get_input<T>());
	my_olc.get_queue()->enqueueWriteBuffer(buffer_weight, CL_TRUE, 0, sizeof(T) * K * N, gemm_ref_ptr->get_weight<T>(g_enable_weight_trans));

	auto kernel_dpp = my_olc.get_kernel();
	kernel_dpp.setArg(0, buffer_intput);
	kernel_dpp.setArg(1, buffer_weight);
	kernel_dpp.setArg(2, buffer_output);
	kernel_dpp.setArg(3, M);
	kernel_dpp.setArg(4, K);
	kernel_dpp.setArg(5, N);
	kernel_dpp.setArg(6, 1.0);
	kernel_dpp.setArg(7, 0.0);

	std::cout << "  == Params:" << std::endl;
	std::cout << "     gws = [" << gws[0] << "," << gws[1]<< "," << gws[2] << "]" << std::endl;
	std::cout << "     lws = [" << lws[0] << "," << lws[1]<< "," << lws[2] << "]" << std::endl;

	// warmup.
	my_olc.get_queue()->enqueueNDRangeKernel(kernel_dpp, cl::NullRange, gws, lws);
	my_olc.get_queue()->finish();

	int64_t total_tm = 0;
	int loop = 10;
	for (int l = 0; l < loop; l++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		my_olc.get_queue()->enqueueNDRangeKernel(kernel_dpp, cl::NullRange, gws, lws);
		my_olc.get_queue()->finish();
		auto t2 = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		total_tm += diff;
		std::cout << "  == tm = " << diff/1000.f << " ms" << std::endl;
	}
	std::cout << "  == mean time = " << (float)total_tm / (loop * 1000.f) << " ms" << std::endl;

	std::cout << "  == Start to copy output from device to host" << std::endl;
	// Copy output from device to host
	my_olc.get_queue()->enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(T) * M * N, output.data());
	std::cout << "      Copy from device to host finish. size = " << output.size() << std::endl;

	return output;
}

void gemm_gpu_opt_block(CGEMM_Ref::Ptr gemm_ref_ptr)
{
	std::cout << "== Test GEMM: " << __FUNCTION__ << std::endl;
	g_dev_inf = get_device_info();

	std::string kernel_fn = "../05_GEMM/src/gemm_kernel_block.cl";
	std::string kernel_entry = "gemm_opt_block";

	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	// =Print some params =================
	auto kernel_perferred_workgroup_size_multiple = get_kernel_perferred_workgroup_size_multiple(my_ocl.get_kernel(), my_ocl.get_device());
	std::cout << "  kernel_perferred_workgroup_size_multiple = " << kernel_perferred_workgroup_size_multiple << std::endl;

	std::cout << "== Start to run GPU kernel : " << kernel_entry << std::endl;
	std::vector<float> outputs_gpu;
	// Only process half data
	{
		auto outputs_gpu_half = run_gemm_kernel<half>(my_ocl, gemm_ref_ptr, kernel_entry);
		outputs_gpu = vec_to_float(outputs_gpu_half);
	}

	// std::cout << "== Ref VS GPU result compare:" << std::endl;
	is_same_buf("== Ref vs GPU gemm:", gemm_ref_ptr->get_output(), outputs_gpu.data(), true ? 0.1 : 0.001, false,
				gemm_ref_ptr->get_m(), gemm_ref_ptr->get_n(), gemm_ref_ptr->get_k());
}