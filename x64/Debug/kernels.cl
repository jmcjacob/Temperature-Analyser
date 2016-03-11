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

__kernel void Max(__global const int* A, __global int* B, __local int* max)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	max[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (max[lid] < max[lid+i])
				max[lid] = max[lid+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid)
	{
		atomic_max(&B[0], max[lid]);
	}
}

__kernel void Min(__global const int* A, __global int* B, __local int* min)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	min[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (min[lid] > min[lid+i])
				min[lid] = min[lid+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid)
	{
		atomic_min(&B[0], min[lid]);
	}
}

__kernel void hist(__global const int* A, __local int* H, __global int* Histogram)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	int min = Histogram[0];
	barrier(CLK_GLOBAL_MEM_FENCE);

	int bin = (A[id]/10) - min;
	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_inc(&H[bin]);
	barrier(CLK_GLOBAL_MEM_FENCE);
	atomic_add(&Histogram[lid+1], H[lid]);
}