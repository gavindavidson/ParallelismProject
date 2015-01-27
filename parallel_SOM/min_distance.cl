__kernel void min_distance(
	__global float * distance_map,
 	const int map_size,
	__global int *winner)
{
	float winner_distance = distance_map[0];
	for (int i = 1; i < map_size; i++){
		if (distance_map[i] < winner_distance){
			winner[0] = i;
			winner_distance = distance_map[i];
		}	
	}
	//winner[0] = winner_index;
}