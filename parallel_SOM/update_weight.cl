__kernel void update_weight(
	__global float *map,
	__global float *input,
	__global float *gauss_value_list,
	__global int *winner_index_array,
	int input_start_index,
	int vector_length,
	int map_side_size//,
	//int *output
	)
{
	int winner_index = winner_index_array[0];
	int current_id = get_global_id(0);
	//output[current_id] = winner_index;
	int a_x, a_y, b_x, b_y;
	a_x = current_id % map_side_size;
	a_y = current_id / map_side_size;
	b_x = winner_index % map_side_size;
	b_y = winner_index / map_side_size;

	int neighbourhood_value = max(abs(a_x-b_x), abs(a_y-b_y));

	int current_map_position;
	for (current_map_position = current_id*vector_length; current_map_position < current_id*vector_length + vector_length; current_map_position++){
		map[current_map_position] = map[current_map_position] - 
					((map[current_map_position] - input[input_start_index + (current_map_position%vector_length)]) * gauss_value_list[neighbourhood_value]);

		//	current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
	}
}