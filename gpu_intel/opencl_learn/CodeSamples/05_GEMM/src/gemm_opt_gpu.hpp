#pragma once
#include "gemm_ref_cpu.hpp"

void gemm_gpu_ref(CGEMM_Ref::Ptr gemm_ref_ptr);

void gemm_gpu_opt_block(CGEMM_Ref::Ptr gemm_ref_ptr);

void gemm_gpu_opt_prepack_b_xmx(CGEMM_Ref::Ptr gemm_ref_ptr);