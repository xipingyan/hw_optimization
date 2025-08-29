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
    // Get work-item and work-group IDs
    const size_t g_id_0 = get_global_id(0);
    const size_t l_id_0 = get_local_id(0);
    const size_t group_id = get_group_id(0);
    const size_t local_ws = get_local_size(0);

    // Calculate the diagonal index for this work-item
    size_t diag_idx = g_id_0;

    // Load a diagonal element into local memory
    // Only work-items with a valid diagonal index will load data
    if (diag_idx < size) {
        local_max_array[l_id_0] = matrix[diag_idx * size + diag_idx];
    } else {
        // Handle out-of-bounds work-items by setting a minimum value
        local_max_array[l_id_0] = -FLT_MAX;
    }

    // Synchronize all work-items in the work-group
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel Reduction using the `__local` memory array
    // This is a single-threaded loop, but with a parallel reduction inside.
    for (size_t s = local_ws / 2; s > 0; s >>= 1) {
        // Only half the work-items participate in each reduction step
        if (l_id_0 < s) {
            local_max_array[l_id_0] = max(local_max_array[l_id_0], local_max_array[l_id_0 + s]);
        }
        // Synchronize after each step to ensure data is updated
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // The work-item with local ID 0 now holds the maximum for the work-group
    if (l_id_0 == 0) {
        // Use an atomic operation to find the global maximum
        // This prevents race conditions when multiple work-groups write to the same output location
        AtomicMax(output, local_max_array[0]);
    }
}

__kernel void dpp_kernel(__global const int *A, __global const int *B, __global int *C)
{
	C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
	printf(" == kernel inside: golbal_id=%zu \n", get_global_id(0));
}