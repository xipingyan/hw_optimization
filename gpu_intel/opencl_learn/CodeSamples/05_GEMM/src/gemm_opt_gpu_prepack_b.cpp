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
static bool g_enable_fp16 = true;
static bool g_enable_weight_trans = true;

cl::Buffer prepack_b(CMyTest& my_olc, CGEMM_Ref::Ptr gemm_ref_ptr) {
	std::string kernel_entry = "gemm_XMX_prepackB";
	auto kernel = my_olc.get_kernel(kernel_entry);

#define WG_SGS 8 // sub-group size (must be 8 for dpasw)
#define WG_N 8	 // max 16
#define WG_M 4	 // max 8
#define SG_M (4 * 8)
#define SG_N (2 * 8)
// all sub-groups in a work-group handles sub-matrix C of shape [BM, BN]
#define BM WG_M *SG_M // 4*(4*8)
#define BN WG_N *SG_N // 8*(2*8)
#define BK 64		  // BK is limited by SLM size
#define SLM_size 65536
#define SLM_use (BM * BK + BN * BK) * 2

	int N = gemm_ref_ptr->get_n();
	int K = gemm_ref_ptr->get_k();

	// [WG_SGS, self.N//SG_N, BM//SG_M], [WG_SGS, WG_N, WG_M]
	auto gws = cl::NDRange(WG_SGS, N/SG_N, BM/SG_M);
	auto lws = cl::NDRange(WG_SGS, WG_N, WG_M);

	auto context = my_olc.get_context();
	cl::Buffer buffer_weight_raw(context, CL_MEM_READ_ONLY, sizeof(half) * N * K);
	cl::Buffer buffer_weight_packed(context, CL_MEM_READ_ONLY, sizeof(half) * K * N);

	// write mat to the device
	my_olc.get_queue()->enqueueWriteBuffer(buffer_weight_raw, CL_TRUE, 0, sizeof(half) * N * K, gemm_ref_ptr->get_weight<half>());

	auto kernel_dpp = my_olc.get_kernel();
	kernel_dpp.setArg(0, buffer_weight_raw);
	kernel_dpp.setArg(1, buffer_weight_packed);
	kernel_dpp.setArg(2, N);
	kernel_dpp.setArg(3, K);
	std::cout << "  == Params prepack_b:" << std::endl;
	std::cout << "     gws = [" << gws[0] << "," << gws[1]<< "," << gws[2] << "]" << std::endl;
	std::cout << "     lws = [" << lws[0] << "," << lws[1]<< "," << lws[2] << "]" << std::endl;

	// warmup.
	my_olc.get_queue()->enqueueNDRangeKernel(kernel_dpp, cl::NullRange, gws, lws);
	my_olc.get_queue()->finish();

	return buffer_weight_packed;
}

// float/fp16
template<typename T>
static std::vector<T> run_gemm_kernel(CMyTest& my_olc, CGEMM_Ref::Ptr gemm_ref_ptr, std::string kernel_entry)
{
	auto M = gemm_ref_ptr->get_m();
	auto N = gemm_ref_ptr->get_n();
	auto K = gemm_ref_ptr->get_k();

	std::vector<T> output(M * N, 0);

	// Reference: https://github.com/usstq/aboutSHW/blob/main/opencl/clops/linear_f16xmx.py#L80
	// Learn how to use intel XMX to accelerate gemm.
#define WG_SGS 8 // sub-group size (must be 8 for dpasw)
#define WG_N 8	 // max 16
#define WG_M 4	 // max 8
#define SG_M (4 * 8)
#define SG_N (2 * 8)
// all sub-groups in a work-group handles sub-matrix C of shape [BM, BN]
#define BM WG_M *SG_M // 4*(4*8)
#define BN WG_N *SG_N // 8*(2*8)
#define BK 64		  // BK is limited by SLM size
#define SLM_size 65536
#define SLM_use (BM * BK + BN * BK) * 2

	auto M_padded = (M + (BM - 1)) / BM * BM;
	auto gws = cl::NDRange(WG_SGS, N/SG_N, M_padded/SG_M); // [WG_SGS, N//SG_N, M_padded//SG_M],
	auto lws = cl::NDRange(WG_SGS, WG_N, WG_M);	   // [WG_SGS, WG_N, WG_M]

	float bias = 0;

	// create buffers on the device
	auto context = my_olc.get_context();
	cl::Buffer buffer_intput(context, CL_MEM_READ_ONLY, sizeof(T) * M * K);
	cl::Buffer buffer_weight_packed = prepack_b(my_olc, gemm_ref_ptr);
	cl::Buffer buffer_bias(context, CL_MEM_READ_WRITE, sizeof(float));
	cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, sizeof(T) * M * N);

	// write mat to the device
	my_olc.get_queue()->enqueueWriteBuffer(buffer_intput, CL_TRUE, 0, sizeof(T) * M * K, gemm_ref_ptr->get_input<T>());
	my_olc.get_queue()->enqueueWriteBuffer(buffer_bias, CL_TRUE, 0, sizeof(float), &bias);

	auto kernel_dpp = my_olc.get_kernel();
	kernel_dpp.setArg(0, buffer_intput);
	kernel_dpp.setArg(1, buffer_weight_packed);
	kernel_dpp.setArg(2, buffer_bias);
	kernel_dpp.setArg(3, buffer_output);
	kernel_dpp.setArg(4, M);
	kernel_dpp.setArg(5, K);
	kernel_dpp.setArg(6, N);

	std::cout << "  == Params :" << kernel_entry << std::endl;
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

// Strange, performance is bad and accuracy is wrong.
void gemm_gpu_opt_prepack_b_xmx(CGEMM_Ref::Ptr gemm_ref_ptr)
{
	std::cout << "== Test GEMM: " << __FUNCTION__ << std::endl;
	g_dev_inf = get_device_info();

	std::string kernel_fn = "../05_GEMM/src/gemm_kernel_opt_with_prepack_b.cl";
	std::string kernel_entry = "gemm_XMX_tput";

	auto my_ocl = CMyTest(kernel_entry, kernel_fn);

	std::cout << "== Start to run GPU kernel : " << kernel_entry << std::endl;
	std::vector<float> outputs_gpu;
	if (g_enable_fp16)
	{
		auto outputs_gpu_half = run_gemm_kernel<half>(my_ocl, gemm_ref_ptr, kernel_entry);
		outputs_gpu = vec_to_float(outputs_gpu_half);
	}

	// std::cout << "== Ref VS GPU result compare:" << std::endl;
	is_same_buf("== Ref vs GPU gemm:", gemm_ref_ptr->get_output(), outputs_gpu.data(), g_enable_fp16 ? 0.1 : 0.001,
				false, gemm_ref_ptr->get_m(), gemm_ref_ptr->get_n(), gemm_ref_ptr->get_k());
}