// __kernel void min_distance(
// 	__global float * distance_map,
//  	const int map_size,
// 	__global int *winner)
// {
// 	float winner_distance = distance_map[0];
// 	for (int i = 1; i < map_size; i++){
// 		if (distance_map[i] < winner_distance){
// 			winner[0] = i;
// 			winner_distance = distance_map[i];
// 		}	
// 	}
// 	//winner[0] = winner_index;
// }

__kernel void min_distance(
	__global float *distance_map,
	__global int *winner_index_array,
	__global float *winner_distance_array,
	const int chunk_size,
	__local float *local_winner_index_array,
	__local float *local_winner_distance_array)
{
	// int current_id = get_global_id(0);
	// int local_winner_index = current_id*chunk_size;
	// int local_winner_distance = distance_map[local_winner_index];
	// for (int i = local_winner_index + 1; i < (current_id+1)*chunk_size; i++){
	// 	if (distance_map[i] == -1){
	// 		break;
	// 	}
	// 	if (local_winner_distance > distance_map[i]){
	// 		local_winner_distance = distance_map[i];
	// 		local_winner_index = i;
	// 	}
	// }
	// winner_distance_array[current_id] = local_winner_distance;
	// winner_index_array[current_id] = local_winner_index;

	// winner_distance_array[0] = 0;
	// winner_index_array[0] = 0;

	int local_chunk_size = (chunk_size / get_local_size(0));
	int start_position = get_global_id(0)*local_chunk_size;
	int local_winner_index = start_position;
	int local_winner_distance = distance_map[local_winner_index];
	for (int i = start_position; i < start_position + local_chunk_size; i++){
		if (distance_map[i] == -1){
			break;
		}
		if(local_winner_distance > distance_map[i]){
			local_winner_distance = distance_map[i];
			local_winner_index = i;
		}
	}
	local_winner_distance_array[get_local_id(0)] = local_winner_distance;
	local_winner_index_array[get_local_id(0)] = local_winner_index;

	barrier(CLK_LOCAL_MEM_FENCE);

	local_winner_index = local_winner_index_array[0];
	local_winner_distance = local_winner_distance_array[0];
	for (int i = 1; i < get_local_size(0); i++){
		if (local_winner_distance > local_winner_distance_array[i]){
			local_winner_distance = local_winner_distance_array[i];
			local_winner_index = local_winner_index_array[i];
		}
	}

	winner_distance_array[get_group_id(0)] = local_winner_distance;
	winner_index_array[get_group_id(0)] = local_winner_index;

	// winner_distance_array[0] = get_group_id(0);
	// winner_index_array[0] = get_group_id(0);
}