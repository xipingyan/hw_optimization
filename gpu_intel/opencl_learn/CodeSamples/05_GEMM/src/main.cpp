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
#include "gemm_ref.hpp"

static size_t g_max_ws_in_one_group[3] = {0};

std::vector<float> run_gemm_kernel(CMyTest& my_olc, CGEMM_Ref::Ptr gemm_ref_ptr, std::string kernel_entry)
{
	auto M = gemm_ref_ptr->get_m();
	auto N = gemm_ref_ptr->get_n();
	auto K = gemm_ref_ptr->get_k();

	std::vector<float> output(M * N, 0);

	// Default reference.
	auto lws = cl::NDRange(1);
	auto gws = cl::NDRange(M, N, 1);
	if (kernel_entry == "gemm_optimized")
	{
#define LWS_SZ 16
		lws = cl::NDRange(LWS_SZ, LWS_SZ, 1);
		gws = cl::NDRange((N + LWS_SZ - 1) / LWS_SZ * LWS_SZ, std::max(LWS_SZ, (M + LWS_SZ - 1) / LWS_SZ * LWS_SZ));
	}

	// create buffers on the device
	auto context = my_olc.get_context();
	cl::Buffer buffer_intput(context, CL_MEM_READ_ONLY, sizeof(float) * M * K);
	cl::Buffer buffer_weight(context, CL_MEM_READ_ONLY, sizeof(float) * K * N);
	cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, sizeof(float) * M * N);

	// write mat to the device
	my_olc.get_queue()->enqueueWriteBuffer(buffer_intput, CL_TRUE, 0, sizeof(float) * M * K, gemm_ref_ptr->get_input());
	my_olc.get_queue()->enqueueWriteBuffer(buffer_weight, CL_TRUE, 0, sizeof(float) * K * N, gemm_ref_ptr->get_weight());

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
	my_olc.get_queue()->enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * M * N, output.data());
	std::cout << "      Copy from device to host finish. size = " << output.size() << std::endl;

	return output;
}

int main()
{
	std::cout << "== Test DPP algorithm. " << std::endl;
	get_device_info(g_max_ws_in_one_group);

	std::string kernel_fn = "../05_GEMM/src/gemm_kernel.cl";
	std::string kernel_entry = "gemm_optimized";
	kernel_entry = "gemm_ref";
	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	// ==================
	std::cout << "== Generate random test data." << std::endl;
	int m = 1, k = 2048, n = 2048;
	m = 3, k = 3584, n = 3584;

	auto gemm_ref = CGEMM_Ref::createPtr(m, n, k);

	std::cout << "== Start to run GPU kernel : " << kernel_entry << std::endl;
	auto outputs_gpu = run_gemm_kernel(my_ocl, gemm_ref, kernel_entry);

	// std::cout << "== Ref VS GPU result compare:" << std::endl;
	is_same_buf("== Ref vs GPU gemm:", gemm_ref->get_output(), outputs_gpu.data(), 0.001, false, m, n, k);

	return 0;
}