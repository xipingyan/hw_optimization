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
        newVal.floatVal = max(prevVal.floatVal,operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

// Methold 1: very poor performance due to atomic contention.
// 对原始的每个元素进行比较，性能较差，每个workitem比较时，都要上锁。
void get_array_max_1(__global const float* input, __local float* local_max_array, __global float* output, const int size)
{
	const size_t global_id = get_global_id(0);
    // printf("    kernel inside: global_id = %d\n", global_id);

	if (global_id < size)
	{
        float value = input[global_id];
        AtomicMax(output, value);
	}
}

// Methold 2: 归约(Reduction), 不彻底的reduction
// local_max_array：每个group内，share同一个地址，把每个group内的max放到local_max_array[0],
// 最终再使用锁，最终结果和每个group内的local_max_array[0]进行比较。
void get_array_max_2(__global const float* input, __local float* local_max_array, __global float* output, const int size)
{
	// Get the work-group ID and the local ID within the work-group.
    const size_t global_id = get_global_id(0);
    const size_t local_id = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t global_ws = get_global_size(0);
    const size_t local_ws = get_local_size(0);

    // printf("global_id = %zu, local_id = %zu, group_id = %zu, global_ws = %zu, local_ws = %zu\n", global_id, local_id, group_id, global_ws, local_ws);

    // 将全局数据加载到本地内存
    local_max_array[local_id] = input[global_id];

    // 同步，确保所有线程都已加载数据
    barrier(CLK_LOCAL_MEM_FENCE);

    // Stage 1: 获取每个subgroup内的最大值
    for (uint s = local_ws / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_max_array[local_id] = max(local_max_array[local_id], local_max_array[local_id+s]);
        }
        // 同步，确保本轮归约完成
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Stage 3: 和每一个subgroup的第一个变量进行比较。最终输出结果output。
    if (local_id == 0) {
        AtomicMax(output, local_max_array[0]);
        // printf("      local_id = %d, %f, \n", local_id, local_max_array[0]);
    }
}


// Methold 3: 基于sub_group_clustered_reduce_max的归约， 测试失败。
void get_array_max_3(__global const float* input, __local float* local_max_array, __global float* output, const int size)
{
	// Get the work-group ID and the local ID within the work-group.
    const size_t global_id = get_global_id(0);
    const size_t local_id = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t global_ws = get_global_size(0);
    const size_t local_ws = get_local_size(0);

    uint sub_group_size = get_sub_group_size();
    uint sub_group_id = local_id / sub_group_size;
    uint sub_group_lid = local_id % sub_group_size;

    printf("global_id = %zu, local_id = %zu, group_id = %zu, global_ws = %zu, local_ws = %zu\n", global_id, local_id, group_id, global_ws, local_ws);

    float my_value = (global_id < size) ? input[global_id] : -FLT_MAX;

    // Stage 1: 获取每个subgroup内的最大值
    float subgroup_max = sub_group_clustered_reduce_max(my_value, sub_group_size);
 
    // Stage 2: Store subgroup maximums in shared local memory
    if (sub_group_lid == 0) {
        local_max_array[sub_group_lid] = subgroup_max;
        printf("  local_max_array[%d] = %f, sub_group_size = %d\n", sub_group_lid, local_max_array[sub_group_lid], sub_group_size);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Stage 3: 将每个subgroup中的
    if (sub_group_id == 0) {
        float group_max = -INFINITY;
        for (uint i = 0; i < (local_ws + sub_group_size - 1) / sub_group_size; i++) {
            if (i * sub_group_size + sub_group_id < local_ws) {
                group_max = max(group_max, local_max_array[i * sub_group_size +sub_group_lid]);
            }
        }
        group_max = sub_group_clustered_reduce_max(group_max, sub_group_size);

        if (sub_group_lid == 0) {
            local_max_array[0] = group_max;
        }
    }

    // Stage 3: 和每一个subgroup的第一个变量进行比较。最终输出结果output。
    if (local_id == 0) {
        AtomicMax(output, local_max_array[0]);
        // printf("      local_id = %d, %f, \n", local_id, local_max_array[0]);
    }
}

__kernel void get_array_max(__global const float* input, 
                            __local float* local_result, 
                            __global float* output, 
                            const int size) {
    // get_array_max_1(input, local_result, output, size);
    get_array_max_2(input, local_result, output, size);
    // get_array_max_3(input, local_result, output, size);
}
