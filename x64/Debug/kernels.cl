__kernel void add(__global const int* A, __global int* B, __local int* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid)
		atomic_add(&B[0], scratch[lid]);
}

__kernel void minMax(__global const int* A, __global int* B, __local int* min, __local int* max)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	min[lid] = A[id];
	max[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (min[id] > min[id+i])
				min[id] = min[id+i];
			if (max[id] < max[id+i])
				max[id] = max[id+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid)
	{
		atomic_min(&B[0], min[lid]);
		atomic_max(&B[1], max[lid]);
	}
}

//__kernel void hist2(__global const int* A, __local int* H )
//{
//	int id = get_global_id(0);
//	int lid = get_local_id(0);
//
//	int bin_index = A[id];
//
//	if (lid < 10)
//		H[lid] = 0;
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	atomic_inc(&H[bin_index(A[id])]);
//}

__kernel void hist(__global const int* A, __global int* H)
{
	int id = get_global_id(0);

	int bin_index = A[id];

	atomic_inc(&H[bin_index]);
}