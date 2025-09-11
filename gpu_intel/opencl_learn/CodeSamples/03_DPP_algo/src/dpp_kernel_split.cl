#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

#define TRANSPOSE_CIS 1

// 跨group，OpenCL本身是不支持同步的，所以求最大值这种操作，必须在一个group内完成。
// Step1: 获取最大支持的group size(一个group最大支持的workitem)，
// Step2: 设置gws和lws到这个group size。
// Step3: 还要传递参数array size，超出group size部分，需要再kernel内部循环。
__kernel void get_array_max_single_group(
    __global const float* input_array, 
    __local float* local_max_array,
    __local int* local_max_ids,
    __global float* output_max,
    __global int* output_id,
    const int array_size
) {
    uint batch_id = get_global_id(0);
    uint lid_0 = get_local_id(1);
    uint local_size = get_local_size(1);

    // 初始化本地最大值为一个极小值
    float my_local_max = -FLT_MAX;
    int my_local_id = 0;
    __global const float* pdata = input_array + batch_id * array_size;

    // --- 阶段1：每个工作项处理它负责的数据块 ---
    // 每个工作项以 local_size 的步长，从全局数组中读取数据
    for (int i = lid_0; i < array_size; i += local_size) {
        if (pdata[i] > my_local_max) {
            my_local_max = pdata[i];
            my_local_id = i;
        }
    }

    // if (batch_id == 0 && lid_0 == 0)
    //     printf("** my_local_id=%d, my_local_max=%f, local_size=%d, lid_0=%d\n", my_local_id, my_local_max, local_size, lid_0);

    // 将每个工作项找到的局部最大值写入共享本地内存
    local_max_array[lid_0] = my_local_max;
    local_max_ids[lid_0] = my_local_id;

    // 同步，确保所有工作项都已完成第一阶段的写入
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 阶段2：并行规约在本地内存中找到最终最大值 ---
    // 这个循环将不断减半，直到 local_max_array[0] 包含最终结果
    for (size_t s = local_size / 2; s > 0; s >>= 1) {
        if (lid_0 < s) {
            // 将当前位置的值与另一个位置的值进行比较
            // printf("  ** local_max_array[%d]=%f > local_max_array[%d]=%f\n", lid_0 + s, local_max_array[lid_0 + s], lid_0, local_max_array[lid_0]);
            if (local_max_array[lid_0 + s] > local_max_array[lid_0] ) {
                local_max_array[lid_0] = local_max_array[lid_0 + s];
                local_max_ids[lid_0] = local_max_ids[lid_0 + s];
            }
            // printf(" ** local_max_array[%d]=%f\n", lid_0, local_max_array[lid_0]);
        }
        // 同步，确保本轮比较完成后，数据对所有线程都可见
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 阶段3：将最终结果写回全局内存 ---
    // 只有本地 ID 为0 的工作项执行此操作
    if (lid_0 == 0) {
        // 将最终结果写入全局输出数组
        output_max[batch_id] = local_max_array[0];
        output_id[batch_id] = local_max_ids[0];
        // if (batch_id == 1)
        //     printf("*output_id = %d, array_size=%d\n", output_id[batch_id], array_size);
    }
}

__kernel void update_orthogonal_vector(__global const float *inp_mat, const int M, __global int *output_id,
                                       int iteration, __global float *cis, __global float *di2s,
                                       const float numerical_threshold, const int selected_token_num)
{
    uint batch_idx = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);

    const int selected_idx = output_id[batch_idx];

    size_t offset = batch_idx * M * M;
    size_t total_tokens = M;
    __global const float *kernel_data = inp_mat + offset;
    __global const float *di2s_data = di2s + batch_idx * M;
    __global float *cis_data = cis + batch_idx * M * selected_token_num;

    // Get the normalization factor
    float norm_factor = sqrt(di2s_data[selected_idx] + numerical_threshold);

    // Compute the new orthogonal vector for each token
    size_t j = gid_1;

    size_t kernel_idx = selected_idx * total_tokens + j;
    float kernel_val = kernel_data[kernel_idx];

    // Subtract the projection onto previously selected vectors
    // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
    float projection = 0.0f;

#if TRANSPOSE_CIS
    __global float *cis_selected_t = cis_data + selected_token_num * selected_idx;
    __global float *cis_t = cis_data + selected_token_num * j;

#if 0 // float4
    int iter4 = iteration / 4;
    int iter_remain = iteration % 4;
    for (size_t prev_t = 0; prev_t < iter4; ++prev_t)
    {
        float4 a_vec = vload4(0, cis_selected_t + prev_t * 4);
        float4 b_vec = vload4(0, cis_t + prev_t * 4);
        projection += dot(a_vec, b_vec);
        // projection += cis_selected_t[prev_t] * cis_t[prev_t];
    }
    for (int prev_t = iter4 - iter_remain; prev_t < iteration; ++prev_t)
    {
        half a_val = cis_selected_t[prev_t];
        half b_val = cis_t[prev_t];
        projection += cis_selected_t[prev_t] * cis_t[prev_t];
    }
#else
    __attribute__((opencl_unroll_hint(4)))
    for (size_t prev_t = 0; prev_t < iteration; ++prev_t)
    {
        projection += cis_selected_t[prev_t] * cis_t[prev_t];
    }
#endif

    // Store the orthogonalized vector element
    size_t cis_current_idx = iteration + j * selected_token_num;
    cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;
#else
    for (size_t prev_t = 0; prev_t < iteration; ++prev_t)
    {
        size_t offset = prev_t * total_tokens;
        size_t cis_selected_idx = offset + selected_idx;
        size_t cis_j_idx = offset + j;
        projection += cis_data[cis_selected_idx] * cis_data[cis_j_idx];
    }
    // Store the orthogonalized vector element
    size_t cis_current_idx = iteration * total_tokens + j;
    cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;
#endif
}

__kernel void update_marginal_gains(const int iteration, const int M, __global int *output_id,
                                    __global float *cis, __global float *di2s,
                                    __global int* output_ids, const int selected_token_num)
{
    uint batch_idx = get_global_id(0);
    uint gid_1 = get_global_id(1);    
    const int selected_idx = output_id[batch_idx];

    __global float *di2s_data = di2s + batch_idx * M;
    __global float *cis_data = cis + batch_idx * M * selected_token_num;
    __global int* output_ids_data = output_ids + batch_idx * selected_token_num;

    uint j = gid_1;
    // Skip updating if this token is already selected (marked as negative infinity)
    if (di2s_data[j] == -INFINITY) {
        return;
    }

#if TRANSPOSE_CIS
    size_t cis_idx = iteration + j * selected_token_num;
#else
    size_t cis_idx = iteration * M + j;
#endif
    float eis_j = cis_data[cis_idx];

    // Subtract the squared orthogonal component
    if (selected_idx == j) {
        di2s_data[selected_idx] = -INFINITY;
        output_ids_data[iteration] = selected_idx;
    }
    else {
        di2s_data[j] -= eis_j * eis_j;
    }
}

__kernel void update_step_2_3(__global const float *inp_mat, const int M, __global int *output_id,
                              int iteration, __global float *cis, __global float *di2s,
                              const float numerical_threshold, const int selected_token_num,
                              __global int *output_ids,
                              __local float *local_max_array,
                              __local int *local_max_ids)
{
    uint batch_idx = get_global_id(0);
    uint gid_1 = get_global_id(1);

    // Step 1: argmax
    uint lid_0 = get_local_id(1);
    uint local_size = get_local_size(1);

    // 初始化本地最大值为一个极小值
    float my_local_max = -FLT_MAX;
    int my_local_id = 0;
    __global const float* pdata = di2s + batch_idx * M;

    for (int i = lid_0; i < M; i += local_size) {
        if (i < M) {
            if (pdata[i] > my_local_max) {
                my_local_max = pdata[i];
                my_local_id = i;
            }
        }
    }

    // 将每个工作项找到的局部最大值写入共享本地内存
    local_max_array[lid_0] = my_local_max;
    local_max_ids[lid_0] = my_local_id;

    // 同步，确保所有工作项都已完成第一阶段的写入
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 阶段2：并行规约在本地内存中找到最终最大值 ---
    // 这个循环将不断减半，直到 local_max_array[0] 包含最终结果
    for (size_t s = local_size / 2; s > 0; s >>= 1) {
        if (lid_0 < s) {
            // 将当前位置的值与另一个位置的值进行比较
            if (local_max_array[lid_0 + s] > local_max_array[lid_0] ) {
                local_max_array[lid_0] = local_max_array[lid_0 + s];
                local_max_ids[lid_0] = local_max_ids[lid_0 + s];
            }
        }
        // 同步，确保本轮比较完成后，数据对所有线程都可见
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Step 2: update orthogonal vector.
    if (gid_1 >= M)
        return;

    const int selected_idx = local_max_ids[0];
    size_t offset = batch_idx * M * M;
    size_t total_tokens = M;
    __global const float *kernel_data = inp_mat + offset;
    __global float *di2s_data = di2s + batch_idx * M;
    __global float *cis_data = cis + batch_idx * M * selected_token_num;
    __global int* output_ids_data = output_ids + batch_idx * selected_token_num;

    // Get the normalization factor
    float norm_factor = sqrt(di2s_data[selected_idx] + numerical_threshold);

    // Compute the new orthogonal vector for each token
    size_t j = gid_1;

    size_t kernel_idx = selected_idx * total_tokens + j;
    float kernel_val = kernel_data[kernel_idx];

    // Subtract the projection onto previously selected vectors
    // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
    float projection = 0.0f;
    __global float *cis_selected_t = cis_data + selected_token_num * selected_idx;
    __global float *cis_t = cis_data + selected_token_num * j;

    __attribute__((opencl_unroll_hint(4))) for (size_t prev_t = 0; prev_t < iteration; ++prev_t)
    {
        projection += cis_selected_t[prev_t] * cis_t[prev_t];
    }

    // Store the orthogonalized vector element
    size_t cis_current_idx = iteration + j * selected_token_num;
    cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;

    // step 3: update_marginal_gains
    size_t cis_idx = iteration + j * selected_token_num;
    float eis_j = cis_data[cis_idx];

    // Subtract the squared orthogonal component
    if (selected_idx == j) {
        di2s_data[selected_idx] = -INFINITY;
        output_ids_data[iteration] = selected_idx;
    }
    else {
        di2s_data[j] -= eis_j * eis_j;
    }
}