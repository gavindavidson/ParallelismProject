#include "drawHTMLMap.h"

int map_side_size, map_vector_size;
float max, min;

float euclidean_distance_from_origin(vector<float> a){
	float sum = 0;
	vector<float>::iterator a_iter = a.begin();
	while (a_iter != a.end()){
		sum += pow((*a_iter), 2);
		a_iter++;
	}
	return sqrt(sum);
}

vector<string> hexColourFromFloat(vector<float> a){
	float sextiles[6];
	float increment = (max - min)/6.0;
	for (int i = 0; i < 6; i++){
		sextiles[i] = i * increment
	}
	string current;
	int current_r_int, current_b_int, current_g_int;
	for (vector<float>::iterator a_iter = a.begin(); a_iter != a.end(); a_iter++){
		current = "#";
		if (*a_iter < sextiles[1]){
			current_r_int = 0;
			current_g_int = 0;
			current_b_int = ((*a_iter)/increment) * 255
		}
		else if (*a_iter < sextiles[2]){
			current_r_int = 0;
			current_g_int = ((*a_iter)-sextiles[1]/increment) * 255;
			current_b_int = 255;
		}
		else if (*a_iter < sextiles[3]){
			current_r_int = 0;
			current_g_int = 255;
			current_b_int = 255 - (((*a_iter)-sextiles[2]/increment) * 255);
		}
		else if (*a_iter < sextiles[4]){
			current_r_int = ((*a_iter)-sextiles[3]/increment) * 255;
			current_g_int = 255;
			current_b_int = 0;
		}
		else{
			current_r_int = 255;
			current_g_int = 255 - (((*a_iter)-sextiles[4]/increment) * 255);
			current_b_int = 0;
		}
		current += decimalToHex(current_r_int) + decimalToHex(current_g_int) + decimalToHex(current_b_int);
	}

}

string decimalToHex(int a){
	std::stringstream ss;
	ss<< std::hex << a; // int decimal_value
	std::string res ( stream.str() );

	return res;
}

void drawMap(vector< vector<float > > map, string filename){
	map_side_size = sqrt((int)map.size());
	map_vector_size = (map.at(0)).size();
	min = FLT_MAX;
	max = FLT_MIN;
	vector<float> weight_map;

	for (vector< vector <float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
		weight_map.push_back(euclidean_distance_from_origin(*map_iter));
	}

	// Determine max and min values for colour purposes
	for (vector< vector <float> >::iterator map_iter = weight_map.begin(); map_iter != weight_map.end(); map_iter++){
			max = fmax(max, *map_iter);
			min = fmin(min, *map_iter);
	}
}