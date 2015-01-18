__kernel void manhattan_distance(
	__global long * subject_vector, 
	__global long * map,
	__global long * distance_array,
	int vector_length)
{
   	int base_map_position = get_global_id(0)*vector_length;
   	
	float sum = 0;
	for (int component = 0; component < vector_length; component++)
	{
		sum += abs(subject_vector[component] - map[base_map_position+component]);	
	}
	distance_array[get_global_id(0)] = sum;
}