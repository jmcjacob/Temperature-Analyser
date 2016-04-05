// Kernel to add all elements in vector together
__kernel void add(__global const int* A, __global int* B, __local int* scratch) 
{
	// Gets IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Copies values to local memory
	scratch[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loops through values and adds them
	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// For each workgroup add total to global
	if (!lid)
		atomic_add(&B[0], scratch[lid]);
}

// Kernel to find max from all elements in vector
__kernel void Max(__global const int* A, __global int* B, __local int* max)
{
	// Gets IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Copies values to local memory
	max[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loops through values to find the max
	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (max[lid] < max[lid+i])
				max[lid] = max[lid+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// For each workgroup find the max
	if (!lid)
	{
		atomic_max(&B[0], max[lid]);
	}
}

// Kernel to find max from all elements in vector
__kernel void Min(__global const int* A, __global int* B, __local int* min)
{
	// Gets IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// Copies values to local memory
	min[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loops through values to find the min
	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (min[lid] > min[lid+i])
				min[lid] = min[lid+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// For each workgroup find the min
	if (!lid)
	{
		atomic_min(&B[0], min[lid]);
	}
}

__kernel void hist(__global const int* A, __local int* H, __global int* Histogram, int nuBins, int max, int min)
{
	// Gets IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int index = A[id];
	
	// Sets values of H to 0
	if (lid < nuBins)
		H[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Skips any padding
	if (index == 999999)
	{
		return;
	}

	// Calculate the bin number to increment
	int bin = (index - min) / ((max-min) / nuBins);

	if (index == max)
		bin--;

	// Increments the local histogram
	
	atomic_inc(&H[bin]);

	barrier(CLK_LOCAL_MEM_FENCE);

	// Combines each local histogram to global histogram
	if (!lid)
	{
		for (int i = 0; i < nuBins; i++)
		{
			atomic_add(&Histogram[i], H[i]);
		}
	}
}