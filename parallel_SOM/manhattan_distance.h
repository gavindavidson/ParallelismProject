#ifndef  STDLIB_H
#include <stdlib.h>
#endif

void manhattan_distance(
	float *subject_vector, 
	float *map, 
	float *distance_array,
	int vector_length,
	int get_global_id);