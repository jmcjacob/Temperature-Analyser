__kernel void minMax(__global const float* input, __global float* output, __local float* maxValue, __local float* minValue)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	
	maxValue[lid] = input[id];
	minValue[lid] = input[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (maxValue[lid] < maxValue[lid + i]) 
				maxValue[lid] = maxValue[lid + i];
			if (minValue[lid] > minValue[lid + i]) 
				minValue[lid] = minValue[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	output[0] = minValue[lid];
	output[1] = maxValue[lid];
}

__kernel void add(__global const float* input, __global float* output, const int inputSize, __local float* sum) 
{
	const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = globalID / localSize;

	sum[localID] = input[globalID];

	for (int offset = localSize / 2; offset > 0; offset /= 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localID < offset)
		{
			sum[localID] += sum[localID + offset];
		}
	}
	if (localID == 0)
	{
		output[workgroupID = sum[0]];
	}
}

__kernel void reduce(__global float* buffer, __local float* scratch, __global float* result) 
{
	int global_index = get_global_id(0);
	int local_index = get_local_id(0);
	int length = get_local_size(0);

	// Load data into local memory
	if (global_index < length) 
	{
		scratch[local_index] = buffer[global_index];
	} 
	else 
	{
		// Infinity is the identity element for the min operation
		scratch[local_index] = INFINITY;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
	{
		if (local_index < offset) 
		{
			float other = scratch[local_index + offset];
			float mine = scratch[local_index];
			scratch[local_index] = (mine < other) ? mine : other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_index == 0) 
	{
		result[get_group_id(0)] = scratch[0];
	}
}