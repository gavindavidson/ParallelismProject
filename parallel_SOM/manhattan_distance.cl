__kernel void manhattan_distance(
	//__global float * subject_vector, 
	__global float * input,
	__global float * map,
	__global float * distance_map,
	int vector_length,
	int map_size,
	int input_start_index)
{
	size_t tid = get_global_id(0);
	if (tid < map_size){
	   	int base_map_position = tid*vector_length;
	   	//int base_map_position = 0;
	   	
		float sum = 0;
		for (int component = 0; component < vector_length; component++)
		{
			sum += fabs(input[input_start_index + component] - map[base_map_position+component]);
			//sum += fabs(subject_vector[component] - map[base_map_position+component]);	
			//sum = fabs(subject_vector[component]);
		}
		distance_map[tid] = sum;
		//distance_map[tid] = 1;
	}
}