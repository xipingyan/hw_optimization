#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

inline void AtomicMax(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = max(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline void AtomicMax_int(volatile __global int *source, const int operand) {
    int prevVal;
    int newVal;
    do {
        prevVal = *source;
        newVal = max(prevVal, operand);
    } while (atomic_cmpxchg((volatile __global int *)source, prevVal, newVal) != prevVal);
}

// low performance
__kernel void get_mat_diagonal_max_1(__global const float* matrix, __local float* local_values, __global float* diagonal_max, const int size)
{
	int gid = get_global_id(1) * get_global_size(2) + get_global_id(2);
	if (gid < size)
	{
		// Each work item handles one element on the diagonal.
        // The diagonal element at index gid is at (gid * size) + gid in a 1D array.
        float value = matrix[gid * size + gid];
        AtomicMax(diagonal_max, value);
	}
}


void AtomicMaxWithID(volatile __global float *maxVal,
                     volatile __global int *maxID,
                     const float operand,
                     const int id) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, prevVal;

    do {
        prevVal.floatVal = *maxVal;
        newVal.floatVal = fmax(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *)maxVal,
                            prevVal.intVal, newVal.intVal) != prevVal.intVal);

    // If we successfully updated the max value, update the ID
    if (newVal.floatVal > prevVal.floatVal) {
        *maxID = id;
    }
}


// Methold 2: 归约(Reduction), 不彻底的reduction
__kernel void get_mat_diagonal_max_2(__global const float* di2s, 
                                     __local float* local_values, __local int* local_ids, 
                                     __global float* best_max_value, __global int* best_max_id,
                                     const int size)
{
    // Get work-item and work-group IDs
    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint lid_0 = get_local_id(0);
    uint lid_1 = get_local_id(1);

    uint group_id = get_group_id(0);
    uint local_ws = get_local_size(1);

    // Load a diagonal element into local memory
    // Only work-items with a valid diagonal index will load data
    if (gid_1 < size) {
        local_values[lid_1] = di2s[gid_1];
        local_ids[lid_1] = gid_1;
    } else {
        // Handle out-of-bounds work-items by setting a minimum value
        local_values[lid_1] = -FLT_MAX;
        local_ids[lid_1] = -1;
    }

    // printf("** gid=%d, lid=%d, value = %f, mat = %f\n", gid_1, lid_1, local_values[lid_1], di2s[gid_1]);

    // Synchronize all work-items in the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel Reduction using the `__local` memory array
    // This is a single-threaded loop, but with a parallel reduction inside.
    for (size_t s = local_ws / 2; s > 0; s >>= 1) {
        // Only half the work-items participate in each reduction step
        if (lid_1 < s) {
            if (local_values[lid_1] < local_values[lid_1 + s]) {
                local_values[lid_1] = local_values[lid_1 + s];
                local_ids[lid_1] = local_ids[lid_1 + s];
            }
        }
        // Synchronize after each step to ensure data is updated
        barrier(CLK_LOCAL_MEM_FENCE);
        // printf(" ** local_ids[%d]=%d\n", lid_1, local_ids[lid_1]);
    }
    
    // The work-item with local ID 0 now holds the maximum for the work-group
    if (lid_1 == 0) {
        // Use an atomic operation to find the global maximum
        // This prevents race conditions when multiple work-groups write to the same best_max_value location
        // AtomicMax(best_max_value, local_values[0]);
        // AtomicMax_int(best_max_id, local_ids[0]);

        AtomicMaxWithID(best_max_value, best_max_id, local_values[0], local_ids[0]);
    }
}

__kernel void find_max_single_group(
    __global const float* input_array, 
    __local float* local_max_array,
    __local int* local_max_ids,
    __global float* output_max,
    __global int* output_id,
    const int array_size
) {
    // 获取当前工作项在工作组内的本地 ID
    const size_t local_id = get_local_id(0);
    // 获取工作组的大小
    const size_t local_size = get_local_size(0);

    // 初始化本地最大值为一个极小值
    float my_local_max = -FLT_MAX;
    int my_local_id = -1;

    // --- 阶段1：每个工作项处理它负责的数据块 ---
    // 每个工作项以 local_size 的步长，从全局数组中读取数据
    for (int i = local_id; i < array_size; i += local_size) {
        if (input_array[i] > my_local_max) {
            my_local_max = input_array[i];
            my_local_id = i;
        }
    }

    // 将每个工作项找到的局部最大值写入共享本地内存
    local_max_array[local_id] = my_local_max;
    local_max_ids[local_id] = my_local_id;

    // 同步，确保所有工作项都已完成第一阶段的写入
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 阶段2：并行规约在本地内存中找到最终最大值 ---
    // 这个循环将不断减半，直到 local_max_array[0] 包含最终结果
    for (size_t s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            // 将当前位置的值与另一个位置的值进行比较
            local_max_array[local_id] = max(local_max_array[local_id], local_max_array[local_id + s]);
        }
        // 同步，确保本轮比较完成后，数据对所有线程都可见
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 阶段3：将最终结果写回全局内存 ---
    // 只有本地 ID 为0 的工作项执行此操作
    if (local_id == 0) {
        // 将最终结果写入全局输出数组
        *output_max = local_max_array[0];
    }
}

void update_orthogonal_vector(__global const float* inp_mat, const int M, const int batch_idx, const int selected_idx, 
                              int iteration, __global float *cis, __global float * di2s,
                              const float numerical_threshold) {
    uint gid_1 = get_global_id(1);

    size_t total_tokens = M;
    float norm_factor = sqrt(di2s[selected_idx] + numerical_threshold);

    size_t kernel_idx = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens + gid_1;
    float kernel_val = inp_mat[kernel_idx];

    float projection = 0.0f;
    // float4
    for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
        size_t cis_selected_idx = prev_t * total_tokens + selected_idx;
        size_t cis_j_idx = prev_t * total_tokens + gid_1;
        projection += cis[cis_selected_idx] * cis[cis_j_idx];
    }

    // Store the orthogonalized vector element
    size_t cis_current_idx = iteration * total_tokens + gid_1;
    cis[cis_current_idx] = (kernel_val - projection) / norm_factor;
    // printf(" cis[%d] = %f\n", cis_current_idx, cis[cis_current_idx]);
}

void update_marginal_gains(const int iteration, const int M, const int selected_idx, 
                           __global float* cis_data, __global float* di2s_data) {

    const int total_tokens = M;

    uint j = get_global_id(1);

    // Skip updating if this token is already selected (marked as negative infinity)
    if (di2s_data[j] == -INFINITY) {
        return;
    }

    size_t cis_idx = iteration * total_tokens + j;
    float eis_j = cis_data[cis_idx];

    // Subtract the squared orthogonal component
    di2s_data[j] -= eis_j * eis_j;
}

__kernel void dpp_kernel(__global const float* inp_mat, __global float *cis, __global float *di2s, 
                         __global int *output_ids, const int batch, const int M, const int selected_num, 
                         __local float* local_values, __local int* local_ids,
                         __global float* best_max_value, __global int* best_max_id,
                         const float numerical_threshold)
{
    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);

    uint gs_0 = get_global_size(0);
    uint gs_1 = get_global_size(1);

    uint lid_0 = get_local_id(0);
    uint lid_1 = get_local_id(1);

    if (gid_1 < M) {
        for (int i = gid_1; i < M; i += gs_1) {
            di2s[i] = inp_mat[i * M + i];
            // printf("** di2s[%d] = %f\n", i, di2s[i]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int i = 0; i < selected_num; i++) {
            // Step 1: Get diagonal max value and id;
            // get_mat_diagonal_max_2(di2s, local_values, local_ids, best_max_value, best_max_id, M);
            find_max_single_group(di2s, local_values, local_ids, best_max_value, best_max_id, M);

            if (gid_1 == 0) {
                output_ids[i] = *best_max_id;
                // printf("  ** kernel inside output_ids[%d] = %d\n", i, output_ids[i]);

               //for (int j = 0; j < selected_num; j++) {
               //    if (output_ids[j] == -1) {
               //        output_ids[j] = *best_max_id;
               //        printf("  ** kernel inside output_ids[%d] = %d\n", j, output_ids[j]);
               //        break;
               //    }
               //}
            }
            barrier(CLK_GLOBAL_MEM_FENCE);

            // step 2
            update_orthogonal_vector(inp_mat, M, gid_0, *best_max_id, i, cis, di2s, numerical_threshold);
            barrier(CLK_GLOBAL_MEM_FENCE);

            // printf("** cis[0][%d] = %f\n", gid_1, cis[gid_1]);

            // Step 3:
            update_marginal_gains(i, M, *best_max_id, cis, di2s);
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

            // Step 4:
            di2s[*best_max_id] = -INFINITY;
            printf(" ** kernel inside: i=%d gid=%d, di2s[%d]%f\n", i, gid_1, *best_max_id, di2s[*best_max_id]);

            *best_max_value = -INFINITY;
            barrier(CLK_GLOBAL_MEM_FENCE);
            // barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        }
        //printf(" ** kernel inside: gid=%d, best_max_id=%d, best_max_value=%f\n", gid_1, *best_max_id, *best_max_value);
    }
}