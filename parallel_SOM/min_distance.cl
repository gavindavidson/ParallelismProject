// Kernel exhibits a two stage reductions that becomes a third stage when
// the host retrieves the two global arrays.
__kernel void min_distance(
	__global float *distance_map,
	__global int *winner_index_array,
	__global float *winner_distance_array,
	const int chunk_size,
	__local float *local_winner_index_array,
	__local float *local_winner_distance_array)
{
	// Processing elements in compute unit produce a local array of possible
	// min distances
	int local_chunk_size = (chunk_size / get_local_size(0));
	int start_position = get_global_id(0)*local_chunk_size;
	int local_winner_index = start_position;
	int local_winner_distance = distance_map[local_winner_index];


	for (int i = start_position; i < start_position + local_chunk_size; i++){
		if (distance_map[i] == -1){
			break;
		}
		else if(local_winner_distance > distance_map[i]){
			local_winner_distance = distance_map[i];
			local_winner_index = i;
		}
	}

	local_winner_distance_array[get_local_id(0)] = local_winner_distance;
	local_winner_index_array[get_local_id(0)] = local_winner_index;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Compute unit iterates through local array to find min distances
	// from this work group and inserts into global array
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
}
