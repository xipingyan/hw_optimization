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
    for (int k = 0; k < K; k++) {
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
    for (int k = 0; k < K; k++) {
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
    for (int k = 0; k < K; k++) {
        tmp += A[i * K + k] * B[j * K + k];
    };
    C[i * N + j] = tmp;
}

// GEMM: M=[1,7], N=2048, K=2048,针对小M，大weight，优化。
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_optimized(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int K, int N,
    float alpha,
    float beta)
{
    const int group_id = get_group_id(0);
    const int local_id = get_local_id(0);
    const int sg_id = get_sub_group_id();
    const int sgl_id = get_sub_group_local_id();
    
    // 针对小m的优化
    const int m = M; // m=1或3
    // if (local_id == 0)
    //    printf("** sg_id = %d, sgl_id = %d\n", sg_id, sgl_id);
    
    // 计算每个工作组处理的n范围
    const int n_per_group = N / get_num_groups(0);
    const int n_start = group_id * n_per_group;
    const int n_end = min(n_start + n_per_group, N);
    
    // 子组内处理16个n元素
    const int n_idx = n_start + sg_id * 16 + sgl_id;
    
    // 私有累加器
    float sum[3] = {0.0f, 0.0f, 0.0f};
    
    // 主循环 - 使用块处理提高数据局部性
    const int block_size = 32;
    for (int k_base = 0; k_base < K; k_base += block_size) {
        int k_end = min(k_base + block_size, K);
        
        // 子组协作加载B矩阵数据
        float b_val = 0.0f;
        if (n_idx < N) {
            for (int k = k_base; k < k_end; k++) {
                b_val = B[k * N + n_idx];
                
                // 处理A矩阵并累加
                for (int mi = 0; mi < m; mi++) {
                    float a_val = A[mi * K + k];
                    sum[mi] += a_val * b_val;
                }
            }
        }
    }
    
    // 写入结果
    if (n_idx < N) {
        for (int mi = 0; mi < m; mi++) {
            int c_index = mi * N + n_idx;
            C[c_index] = alpha * sum[mi] + beta * C[c_index];
        }
    }
}