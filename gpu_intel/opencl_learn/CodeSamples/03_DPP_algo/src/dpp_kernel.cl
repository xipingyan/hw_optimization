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
    const size_t lid_0 = get_local_id(0);
    // 获取工作组的大小
    const size_t local_size = get_local_size(0);

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
    }
}

void update_orthogonal_vector(__global const float* inp_mat, const int M, const int batch_idx, const int selected_idx, 
                              int iteration, __global float *cis, __global float * di2s,
                              const float numerical_threshold) {
    uint gid_1 = get_global_id(1);
    uint gs_1 = get_global_size(1);

#if 0
    size_t total_tokens = M;

    float norm_factor = sqrt(di2s[selected_idx] + numerical_threshold);
    float inv_norm = 1.0f / norm_factor;

    size_t base_kernel_offset = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens;

    // float *cis_out = cis + iteration * total_tokens;
    size_t cis_offset = iteration * total_tokens;

    for (int m_idx = gid_1; m_idx < M; m_idx += gs_1) {
        cis[cis_offset + m_idx] = inp_mat[base_kernel_offset + m_idx];
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int m_idx = gid_1; m_idx < M; m_idx += gs_1) {
        int prev_t = m_idx;
        if (prev_t < iteration) {
            size_t cis_prev_row_offset = prev_t * total_tokens;
            float cis_sel = cis[cis_prev_row_offset + selected_idx];
            if (fabs(cis_sel) < 1e-10f) {
                continue;
            }

#if 0
            float4 data_in1 = {cis_sel,cis_sel,cis_sel,cis_sel};
            for (size_t j = 0; j < total_tokens/4; ++j)
            {
                float4 data_in2 = vload4(0, &cis[cis_prev_row_offset + j]);
                float4 dst = vload4(0, &cis[cis_offset + j]);
                dst -= data_in1 * data_in2;
                vstore4(dst, 0, &cis[cis_offset + j]); 
            }
#else 
            for (size_t j = 0; j < total_tokens; ++j) {
                cis[cis_offset + j] -= cis_sel * cis[cis_prev_row_offset + j];
            }
#endif
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int m_idx = gid_1; m_idx < M; m_idx += gs_1)
    {
        cis[cis_offset + m_idx] *= inv_norm;
    }
#else
    size_t total_tokens = M;
    float norm_factor = sqrt(di2s[selected_idx] + numerical_threshold);
    size_t m_offset = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens;
    for (int m_idx = gid_1; m_idx < M; m_idx += gs_1) {
       
        size_t kernel_idx = m_offset + m_idx;
        float kernel_val = inp_mat[kernel_idx];
    
        float projection = 0.0f;
        for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
            size_t tmp_offset = prev_t * total_tokens;
            size_t cis_selected_idx = tmp_offset + selected_idx;
            size_t cis_j_idx = tmp_offset + m_idx;
            projection += cis[cis_selected_idx] * cis[cis_j_idx];
        }
    
        // Store the orthogonalized vector element
        size_t cis_current_idx = iteration * total_tokens + m_idx;
        cis[cis_current_idx] = (kernel_val - projection) / norm_factor;
    }
#endif
}

void update_marginal_gains(const int iteration, const int selected_num, const int selected_idx, 
                           __global float* cis_data, __global float* di2s_data) {
    uint gid_1 = get_global_id(1);
    uint gs_1 = get_global_size(1);

    for (int m_idx = gid_1; m_idx < selected_num; m_idx += gs_1) {
        // Skip updating if this token is already selected (marked as negative infinity)
        if (di2s_data[m_idx] == -INFINITY) {
            return;
        }

        size_t cis_idx = iteration * selected_num + m_idx;
        float eis_j = cis_data[cis_idx];

        // Subtract the squared orthogonal component
        di2s_data[m_idx] -= eis_j * eis_j;
        
        // printf("  *** di2s_data[%d] = %f, eis_j = %f, cis_idx = %d\n", m_idx, di2s_data[m_idx], eis_j, cis_idx);
    }
}

// Single group dpp kernel.
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
        // Copy diagonal values to di2s
        for (int m_idx = gid_1; m_idx < M; m_idx += gs_1) {
            di2s[m_idx] = inp_mat[m_idx * M + m_idx];
            // printf("** di2s[%d] = %f\n", m_idx, di2s[m_idx]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int i = 0; i < selected_num; i++) {
            // Step 1: Get diagonal max value and id;
            get_array_max_single_group(di2s, local_values, local_ids, best_max_value, best_max_id, M);
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            if (gid_1 == 0) {
                // printf("** loop[%d] best_max_value=%f, best_max_id=%d\n", i, *best_max_value, *best_max_id);

                output_ids[i] = *best_max_id;
                // printf("  ** kernel inside output_ids[%d] = %d\n", i, output_ids[i]);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);

            // step 2
            update_orthogonal_vector(inp_mat, M, gid_0, *best_max_id, i, cis, di2s, numerical_threshold);
            barrier(CLK_GLOBAL_MEM_FENCE);

            for (int m_idx = gid_1; m_idx < M; m_idx += gs_1) {
                // printf("** cis[%d][%d]=%f\n", i, m_idx, cis[i*M + m_idx]);
            }

            // Step 3:
            update_marginal_gains(i, selected_num, *best_max_id, cis, di2s);
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

            //for (int m_idx = gid_1; m_idx < M; m_idx += gs_1) {
            //    printf("  ** di2s[%d][%d]=%f\n", i, m_idx, di2s[i*M + m_idx]);
            //}

            // Step 4:
            di2s[*best_max_id] = -INFINITY;
            // printf(" ** kernel inside: i=%d gid=%d, di2s[%d]%f\n", i, gid_1, *best_max_id, di2s[*best_max_id]);

            *best_max_value = -INFINITY;
            barrier(CLK_GLOBAL_MEM_FENCE);
            // barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        }
        //printf(" ** kernel inside: gid=%d, best_max_id=%d, best_max_value=%f\n", gid_1, *best_max_id, *best_max_value);
    }
}