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
    // printf("    kernel inside: gid_0 = %d\n", gid_0);

	if (gid_0 < size)
	{
        float value = input[gid_0];
        AtomicMax(output, value);
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

    // 同步，确保所有线程都已加载数据
    barrier(CLK_LOCAL_MEM_FENCE);

    // Stage 1: 获取每个subgroup内的最大值
    for (uint s = l_ws_0 / 2; s > 0; s >>= 1) {
        if (lid_0 < s) {
            local_max_array[lid_0] = max(local_max_array[lid_0], local_max_array[lid_0+s]);
        }
        // 同步，确保本轮归约完成
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Stage 3: 和每一个subgroup的第一个变量进行比较。最终输出结果output。
    if (lid_0 == 0) {
        AtomicMax(output, local_max_array[0]);
        printf("      lid_0 = %d, %f, \n", lid_0, local_max_array[0]);
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
    int my_local_id = -1;

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

// Methold 3: 基于sub_group_clustered_reduce_max的归约， 测试失败。
void get_array_max_3(__global const float* input, __local float* local_max_array, __global float* output, const int size)
{
	// Get the work-group ID and the local ID within the work-group.
    const size_t gid_0 = get_global_id(0);
    const size_t lid_0 = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t g_ws_0 = get_global_size(0);
    const size_t l_ws_0 = get_local_size(0);

    uint sub_group_size = get_sub_group_size();
    uint sub_group_id = lid_0 / sub_group_size;
    uint sub_group_lid = lid_0 % sub_group_size;

    printf("gid_0 = %zu, lid_0 = %zu, group_id = %zu, g_ws_0 = %zu, l_ws_0 = %zu\n", gid_0, lid_0, group_id, g_ws_0, l_ws_0);

    float my_value = (gid_0 < size) ? input[gid_0] : -FLT_MAX;

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
        for (uint i = 0; i < (l_ws_0 + sub_group_size - 1) / sub_group_size; i++) {
            if (i * sub_group_size + sub_group_id < l_ws_0) {
                group_max = max(group_max, local_max_array[i * sub_group_size +sub_group_lid]);
            }
        }
        group_max = sub_group_clustered_reduce_max(group_max, sub_group_size);

        if (sub_group_lid == 0) {
            local_max_array[0] = group_max;
        }
    }

    // Stage 3: 和每一个subgroup的第一个变量进行比较。最终输出结果output。
    if (lid_0 == 0) {
        AtomicMax(output, local_max_array[0]);
        // printf("      lid_0 = %d, %f, \n", lid_0, local_max_array[0]);
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
