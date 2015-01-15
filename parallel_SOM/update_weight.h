#ifndef  STDLIB_H
#include <stdlib.h>
#include <math.h>
#endif

void update_weight(
	float *map,
	float *input,
	float *gauss_value_list,
	int winner_index,
	int input_start_index,
	int vector_length,
	int map_side_size,
	int current_id		// ONLY FOR NON OPENCL VERSION
	);