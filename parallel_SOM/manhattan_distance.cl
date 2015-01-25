__kernel void manhattan_distance(
	__global float * input,
	__global float * map,
	__global float * distance_map,
	int vector_length,
	int input_start_index)
{
	size_t tid = get_global_id(0);
   	int base_map_position = tid*vector_length;
   	
	float sum = 0;
	for (int component = 0; component < vector_length; component++)
	{
		sum += fabs(input[input_start_index + component] - map[base_map_position+component]);
	}
	distance_map[tid] = sum;
}