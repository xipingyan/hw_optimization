__kernel void get_diagonal_max(__global const float* matrix, __global float* diagonal_max, const int size)
{
	int gid = get_global_id(0);
	if (gid < size)
	{
		// Each work item handles one element on the diagonal.
        // The diagonal element at index gid is at (gid * size) + gid in a 1D array.
        float value = matrix[gid * size + gid];
        
        // This is a naive approach. A proper reduction kernel is needed
        // for parallel max-finding. For simplicity, we'll store the
        // diagonal elements and find the max on the host.
        // A more complex reduction kernel would be more efficient for large matrices.
        
        // This simplified approach assumes the host will do the final reduction.
        // We'll just write the diagonal elements to a temporary buffer.
        // A more advanced solution would use a parallel reduction algorithm.
        
        // For a demonstration, let's just use an atomic operation, which is very inefficient
        // for this purpose but illustrates a concept. A real-world solution would
        // use a proper reduction algorithm.
        
        // This is a simplified, demonstrative approach.
        // It's not the most efficient but shows the basic idea.
        // A proper reduction would involve multiple work groups and shared memory.
        
        // Naive but illustrative approach
        // Each work item attempts to update the global max value.
        // This will have very poor performance due to atomic contention.
        // A proper reduction would be more complex and efficient.
        
        atomic_max(diagonal_max, value);
	}
}

void kernel dpp_kernel(global const int *A, global const int *B, global int *C)
{
	C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
	printf(" == kernel inside: golbal_id=%zu \n", get_global_id(0));
}