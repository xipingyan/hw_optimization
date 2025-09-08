#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

__kernel void gemm_ref(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int K, int N,
    float alpha,
    float beta)
{
    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);

    uint i = gid_0;
    uint j = gid_1;

    float tmp = 0;
    // 让编译器尝试着去展开循环，性能提升很大
    __attribute__((opencl_unroll_hint(8))) for (int k = 0; k < K; k++)
    {
        tmp += A[i * K + k] * B[k * N + j];
    };
    C[i * N + j] = tmp;
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm_ref_half(
    __global const half *A,
    __global const half *B,
    __global half *C,
    int M, int K, int N,
    float alpha,
    float beta)
{
    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);

    uint i = gid_0;
    uint j = gid_1;

    float tmp = 0;
    for (int k = 0; k < K; k++)
    {
        tmp += A[i * K + k] * B[k * N + j];
    };
    C[i * N + j] = tmp;
}

__kernel void gemm_ref_half_weight_trans(
    __global const half *A,
    __global const half *B,
    __global half *C,
    int M, int K, int N,
    float alpha,
    float beta)
{
    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);

    uint i = gid_0; // M
    uint j = gid_1; // N

    float tmp = 0;
    __attribute__((opencl_unroll_hint(8))) for (int k = 0; k < K; k++)
    {
        tmp += A[i * K + k] * B[j * K + k];
    };
    C[i * N + j] = tmp;
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm_half4_weight_trans(
    __global const half *A,
    __global const half *B,
    __global half *C,
    int M, int K, int N,
    float alpha,
    float beta)
{
    int i = get_global_id(0); // M
    int j = get_global_id(1); // N

    if (i >= M || j >= N)
        return;

    float acc = 0.0f;

    int k4 = K / 4;
    int k_remain = K % 4;

    int off_i = i * K;
    int off_j = j * K;

    for (int k = 0; k < k4; ++k)
    {
        int a_idx = off_i + k * 4;
        int b_idx = off_j + k * 4;

        // Read half4, return float4.
        float4 a_vec = vload_half4(0, A + a_idx);
        float4 b_vec = vload_half4(0, B + b_idx);

        acc += dot(a_vec, b_vec);
    }

    // 处理剩余的 K % 4
    for (int k = K - k_remain; k < K; ++k)
    {
        half a_val = A[i * K + k];
        half b_val = B[j * K + k];
        acc += convert_float(a_val) * convert_float(b_val);
    }

    C[i * N + j] = convert_half(acc);
}