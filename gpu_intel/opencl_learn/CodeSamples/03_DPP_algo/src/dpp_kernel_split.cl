#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable


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
    // 获取当前工作项在工作组内的本地 ID
    const size_t lid_0 = get_local_id(1);
    // 获取工作组的大小
    const size_t local_size = get_local_size(1);

    // 初始化本地最大值为一个极小值
    float my_local_max = -FLT_MAX;
    int my_local_id = 0;

    // --- 阶段1：每个工作项处理它负责的数据块 ---
    // 每个工作项以 local_size 的步长，从全局数组中读取数据
    for (int i = lid_0; i < array_size; i += local_size) {
        if (input_array[i] > my_local_max) {
            my_local_max = input_array[i];
            my_local_id = i;
        }
    }
    // printf("** my_local_id=%d, my_local_max=%f, local_size=%d, lid_0=%d\n", my_local_id, my_local_max, local_size, lid_0);

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
        *output_max = local_max_array[0];
        *output_id = local_max_ids[0];
        // printf("*output_id = %d, array_size=%d\n", *output_id, array_size);
    }
}

__kernel void update_orthogonal_vector(__global const float *inp_mat, const int M, __global int *output_id,
                                       int iteration, __global float *cis, __global float *di2s,
                                       const float numerical_threshold)
{
    uint batch_idx = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);

    const int selected_idx = output_id[0];

    size_t total_tokens = M;
    __global const float *kernel_data = inp_mat;
    __global const float *di2s_data = di2s;
    __global float *cis_data = cis;

    // Get the normalization factor
    float norm_factor = sqrt(di2s_data[selected_idx] + numerical_threshold);

    // Compute the new orthogonal vector for each token
    size_t j = gid_1;

    size_t kernel_idx = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens + j;
    float kernel_val = kernel_data[kernel_idx];

    // Subtract the projection onto previously selected vectors
    // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
    float projection = 0.0f;
    for (size_t prev_t = 0; prev_t < iteration; ++prev_t)
    {
        size_t cis_selected_idx = prev_t * total_tokens + selected_idx;
        size_t cis_j_idx = prev_t * total_tokens + j;
        projection += cis_data[cis_selected_idx] * cis_data[cis_j_idx];
    }

    // Store the orthogonalized vector element
    size_t cis_current_idx = iteration * total_tokens + j;
    cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;
}

__kernel void update_marginal_gains(const int iteration, const int M, __global int *output_id,
                                    __global float *cis_data, __global float *di2s_data,
                                    __global int* buffer_output_ids)
{
    uint gid_1 = get_global_id(1);
    
    const int selected_idx = output_id[0];
 
    uint j = gid_1;
    // Skip updating if this token is already selected (marked as negative infinity)
    if (di2s_data[j] == -INFINITY) {
        return;
    }

    size_t cis_idx = iteration * M + j;
    float eis_j = cis_data[cis_idx];

    // Subtract the squared orthogonal component
    if (selected_idx == j) {
        di2s_data[selected_idx] = -INFINITY;
        buffer_output_ids[iteration] = selected_idx;
    }
    else
        di2s_data[j] -= eis_j * eis_j;
}