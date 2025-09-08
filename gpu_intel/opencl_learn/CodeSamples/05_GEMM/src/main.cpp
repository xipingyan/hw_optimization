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
#include "gemm_ref_cpu.hpp"
#include "gemm_opt_gpu.hpp"

int main()
{
	std::cout << "== Test GEMM algorithm. " << std::endl;

	std::cout << "== Generate random test data." << std::endl;
	int m = 1, k = 2048, n = 2048;
	m = 3, k = 3584, n = 3584;// [3584, 4608, 18944]

	get_env_int("M", m);
	get_env_int("N", n);
	get_env_int("K", k);

	auto gemm_ref_ptr = CGEMM_Ref::createPtr(m, n, k);

	if (get_env_bool("GPU_REF"))
	{
		gemm_gpu_ref(gemm_ref_ptr);
	}
	else if (get_env_bool("GPU_BLOCK"))
	{
		gemm_gpu_opt_block(gemm_ref_ptr);
	}
	else if (get_env_bool("GPU_PREPACK_B"))
	{
		gemm_gpu_opt_prepack_b_xmx(gemm_ref_ptr);
	}

	return 0;
}