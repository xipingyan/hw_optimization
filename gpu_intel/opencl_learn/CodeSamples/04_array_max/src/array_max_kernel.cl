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
	const size_t gid_0 = get_global_id(0);
    const size_t group_id = get_group_id(0);

    // printf("    kernel inside: gid_0 = %d\n", gid_0);

	if (gid_0 < size)
	{
        float value = input[gid_0];
        // 每个group，求一个最大值，返回host后，host在计算所有group的值。
        AtomicMax(&output[group_id], value);
	}
}

// Methold 2: 归约(Reduction), 不彻底的reduction
// local_max_array：每个group内，share同一个地址，把每个group内的max放到local_max_array[0],
// 最终再使用锁，最终结果和每个group内的local_max_array[0]进行比较。
void get_array_max_2(__global const float* input, __local float* local_max_array, __global float* output, const int size)
{
	// Get the work-group ID and the local ID within the work-group.
    const size_t gid_0 = get_global_id(0);
    const size_t lid_0 = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t g_ws_0 = get_global_size(0);
    const size_t l_ws_0 = get_local_size(0);

    // printf("gid_0 = %zu, lid_0 = %zu, group_id = %zu, g_ws_0 = %zu, l_ws_0 = %zu\n", gid_0, lid_0, group_id, g_ws_0, l_ws_0);
    if (gid_0 >= size) {
        return;
    }
    local_max_array[lid_0] = input[gid_0];

    // 同步，确保当前EU的所有线程都已加载数据
    barrier(CLK_LOCAL_MEM_FENCE);

    // Stage 1: 获取每个subgroup内的最大值
    for (uint s = l_ws_0 / 2; s > 0; s >>= 1) {
        if (lid_0 < s) {
            local_max_array[lid_0] = max(local_max_array[lid_0], local_max_array[lid_0+s]);
        }
        // 同步，确保本轮归约完成
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Stage 3: 和每一个group的第一个变量进行比较。最终把当前group的最大值输出结果output。
    if (lid_0 == 0) {
        AtomicMax(&output[group_id], local_max_array[0]);
        // printf("      lid_0 = %d, %f, \n", lid_0, local_max_array[0]);
    }
}

// Methold 3: 基于sub_group_clustered_reduce_max的归约
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void get_array_max_3(__global const float* input, __local float* local_max_array, __global float* output, const int size)
{
	// Get the work-group ID and the local ID within the work-group.
    const size_t gid_0 = get_global_id(0);
    const size_t lid_0 = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t g_ws_0 = get_global_size(0);
    const size_t l_ws_0 = get_local_size(0);

    uint sub_group_size = get_sub_group_size();
    uint sub_group_id = get_sub_group_id();
    uint sub_group_lid = get_sub_group_local_id();

    uint sub_groups_num = l_ws_0 / sub_group_size;

    // printf("gid_0 = %zu, lid_0 = %zu, group_id = %zu, g_ws_0 = %zu, l_ws_0 = %zu\n", gid_0, lid_0, group_id, g_ws_0, l_ws_0);
    // printf("sub_group_size = %zu, sub_group_id = %zu, sub_group_lid=%zu, l_ws_0=%zu\n", sub_group_size, sub_group_id, sub_group_lid, l_ws_0);

    float my_value = (gid_0 < size) ? input[gid_0] : -FLT_MAX;

    // Stage 1: 获取每个subgroup内的最大值
    float sub_group_max = sub_group_reduce_max(my_value);
    if (sub_group_lid == 0) {
        local_max_array[sub_group_id] = sub_group_max;
        // printf("%f, sub_group_id=%d\n", local_max_array[sub_group_id], sub_group_id);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t s = sub_groups_num / 2; s > 0; s >>= 1) {
        if (lid_0 < s) {
            if (local_max_array[lid_0 + s] > local_max_array[lid_0] ) {
                local_max_array[lid_0] = local_max_array[lid_0 + s];
            }
        }
        // 同步，确保本轮比较完成后，数据对所有线程都可见
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid_0 == 0) {
        output[group_id] = local_max_array[0];
    }
}

__kernel void get_array_max(__global const float* input, 
                            __local float* local_result, 
                            __global float* output, 
                            const int size) {
    // get_array_max_1(input, local_result, output, size);
    // get_array_max_2(input, local_result, output, size);
    get_array_max_3(input, local_result, output, size);
}