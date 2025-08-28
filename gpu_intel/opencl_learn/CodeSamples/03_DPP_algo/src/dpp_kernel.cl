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

// low performance
__kernel void get_mat_diagonal_max_1(__global const float* matrix, __local float* local_max_array, __global float* diagonal_max, const int size)
{
	int gid = get_global_id(0);
	if (gid < size)
	{
		// Each work item handles one element on the diagonal.
        // The diagonal element at index gid is at (gid * size) + gid in a 1D array.
        float value = matrix[gid * size + gid];
        AtomicMax(diagonal_max, value);
	}
}

// Methold 2: 归约(Reduction), 不彻底的reduction
// local_max_array：每个group内，share同一个地址，把每个group内的max放到local_max_array[0],
// 最终再使用锁，最终结果和每个group内的local_max_array[0]进行比较。
__kernel void get_mat_diagonal_max_2(__global const float* matrix, __local float* local_max_array, __global float* output, const int size)
{
	// Get the work-group ID and the local ID within the work-group.
    const size_t global_id = get_global_id(0);
    const size_t local_id = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t global_ws = get_global_size(0);
    const size_t local_ws = get_local_size(0);

    // 将全局数据加载到本地内存
    if (global_id >= size) {
        return;
    }
    // printf("global_id = %zu, local_id = %zu, group_id = %zu, global_ws = %zu, local_ws = %zu\n", global_id, local_id, group_id, global_ws, local_ws);

    uint idx = global_id * size + global_id;
    local_max_array[local_id] = matrix[idx];

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

__kernel void dpp_kernel(__global const int *A, __global const int *B, __global int *C)
{
	C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
	printf(" == kernel inside: golbal_id=%zu \n", get_global_id(0));
}