#include "manhattan_distance.h"

void manhattan_distance(
	float *subject_vector, 
	float *map,
	float *distance_array,
	int vector_length,
	int get_global_id){				// get_global_id will be removed when this is a kernal and replaced by the 'get_global_id(0)' openCL function


   	//int base_map_position = get_global_id(0)*vector_length;
   	int base_map_position = get_global_id*vector_length;

	float sum = 0;
	int component;
	for (component = 0; component < vector_length; component++){
		sum += abs(subject_vector[component] - map[base_map_position+component]);
	}
	distance_array[get_global_id] = sum;
}
