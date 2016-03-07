void atomic_add_global(volatile global float *source, const float operand) {
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
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void minMax(__global const float* A, __global float* B, __global float* C)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];
	C[id] = A[id];
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int stride = N/2; stride >= 1; stride /= 2)
	{
		if (!(id % (stride * 2)) && ((id + stride) < N))
		{
			B[id] = fmin(B[id], B[id+stride]);
			C[id] = fmax(C[id], C[id+stride]);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

}

__kernel void add(__global float* A, __global float* B) 
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 1; i < N; i *= 2) {
		if (!(id % (i * 2)) && ((id + i) < N)) 
			A[id] += A[id + i];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	B[id] = A[id];
}