#include "update_weight.h"

void update_weight(
	float *map,
	float *input,
	float *gauss_value_list,
	int winner_index,
	int input_start_index,
	int vector_length,
	int map_side_size,
	int current_id		// ONLY FOR NON OPENCL VERSION
	){

	// int map_index = get_global_id(0);		FOR OPENCL VERSION
	// neighbourhood_value = determineSteps(map_index/vector_size, winner_index);

	int a_x, a_y, b_x, b_y, output;
	a_x = current_id % map_side_size;
	a_y = current_id / map_side_size;
	b_x = winner_index % map_side_size;
	b_y = winner_index / map_side_size;

	int neighbourhood_value = fmax(abs(a_x-b_x), abs(a_y-b_y));

	int current_map_position;
	for (current_map_position = current_id*vector_length; current_map_position < current_id*vector_length + vector_length; current_map_position++){
		map[current_map_position] = map[current_map_position] - 
					((map[current_map_position] - input[input_start_index + (current_map_position%vector_length)]) * gauss_value_list[neighbourhood_value]);
		/*
			current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
		*/
	}
}