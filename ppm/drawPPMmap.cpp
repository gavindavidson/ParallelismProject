#include "drawPPMmap.h"

int map_side_size, map_vector_size, map_size;
float max_weight, min_weight;

string height = "4";
string width = "4"; 	//For html output

float euclidean_distance_from_origin(vector<float> a){
	float sum = 0;
	vector<float>::iterator a_iter = a.begin();
	while (a_iter != a.end()){
		sum += pow((*a_iter), 2);
		a_iter++;
	}
	return sqrt(sum);
}

string decimalToHex(int a){
	std::stringstream ss;
	ss<< std::hex << a; // int decimal_value
	std::string res ( ss.str() );
	if (a < 16){		// Because of hexidecimal and the fact that colours must be of the format "#RRGGBB"
		res = "0" + res;
	}
	return res;
}

RGB *rgbColourFromFloat(float* a){
	float sextiles[6];
	float increment = (max_weight - min_weight)/6.0;
	// vector<string> output;
	RGB *output = (RGB*)malloc(sizeof(RGB) * map_size);
	for (int i = 0; i < 6; i++){
		sextiles[i] = i * increment + min_weight;
	}
	int current_r_int, current_b_int, current_g_int;
	//for (vector<float>::iterator a_iter = a.begin(); a_iter != a.end(); a_iter++){
	for (int i = 0; i < map_size; i++){
		if (a[i] < sextiles[1]){
			current_r_int = 0;
			current_g_int = 0;
			current_b_int = ((a[i]-sextiles[0])/increment) * 255;
			//std::cout << "sextiles[1]\t";// << std::endl;
		}
		else if (a[i] < sextiles[2]){
			current_r_int = 0;
			current_g_int = ((a[i]-sextiles[1])/increment) * 255;
			current_b_int = 255;
			//std::cout << "sextiles[2]\t";// << std::endl;
		}
		else if (a[i] < sextiles[3]){
			current_r_int = 0;
			current_g_int = 255;
			current_b_int = 255 - (((a[i]-sextiles[2])/increment) * 255);
			// std::cout << "sextiles[3]\t";// << std::endl;
		}
		else if (a[i] < sextiles[4]){
			current_r_int = ((a[i]-sextiles[3])/increment) * 255;
			current_g_int = 255;
			current_b_int = 0;
			// std::cout << "sextiles[4]\t";// << std::endl;
		}
		else if (a[i] < sextiles[5]){
			current_r_int = 255;
			current_g_int = 255 - (((a[i]-sextiles[4])/increment) * 255);
			current_b_int = 0;
			// std::cout << "sextiles[5]\t";// << std::endl;
		}
		else {
			current_r_int = 255;
			current_g_int = 0;
			current_b_int = ((a[i]-sextiles[5])/increment) * 255;
			// std::cout << "sextiles[6]\t";// << std::endl;
		}
		if (current_b_int > 255 && a[i] < sextiles[1]){
			std::cout << "VAL:" << a[i] << "\tMAX:" << max_weight << 
				"\t" << current_r_int << ", " << current_g_int << ", " << current_b_int << std::endl;
			std::cout << "SEXTILES: ";
			for (int i = 0; i < 6; i++){
				std::cout << sextiles[i] << ",\t";
			}
			std::cout << std::endl;
		}

		// if (current_r_int > 255){
		// 	std::cout << "current_r_int: " << current_r_int;
		// } if (current_g_int > 255){
		// 	std::cout << "current_g_int: " << current_g_int;
		// } if (current_b_int > 255){
		// 	std::cout << "current_b_int: " << current_b_int;
		// }

		//std::cout << std::endl;
		output[i].red = current_r_int;
		output[i].green = current_g_int;
		output[i].blue = current_b_int;
		// output.push_back(current);
	}
	return output;
}

// string produceHTML(vector<string> hex_colours){
// 	string output = "<table cellspacing=\"0\">\n<tr>";
// 	int current_index = 0;
// 	for (vector<string>::iterator colour_iter = hex_colours.begin(); colour_iter != hex_colours.end(); colour_iter++){
// 		if (current_index%map_side_size == 0){
// 			output += "\n</tr>\n<tr>";
// 		}
// 		output += string("\t<td style=\"width:") + width + "px;height:" + height + "px;background-color:" + (*colour_iter) + "\">";
// 		current_index++;
// 	}
// 	output += "</tr>\n</table>";
// 	return output;
// }

int writeToFile(string contents, string filename){
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << contents;
	outfile.close();
	return 0;
}

int drawMap(float *map, int in_map_size, int vector_length, string filename){
	map_size = in_map_size;
	map_side_size = sqrt(map_size);
	map_vector_size = vector_length;
	vector<float> current_vector;
	float *weight_map = (float *)malloc(sizeof(float) * map_size);
	min_weight = FLT_MAX;
	max_weight = FLT_MIN;
	for (int i = 0; i < map_size*vector_length; i++){
		if (i%vector_length == 0 && i!=0){
			weight_map[i/vector_length - 1] = euclidean_distance_from_origin(current_vector);
			current_vector.clear();
		}
		current_vector.push_back(map[i]);
	}
	// Determine max and min weights for colour values for the sake of scale
	for (int i = 0; i < map_size; i++){
		if (max_weight < weight_map[i]){
			max_weight = weight_map[i];
		}
		else if (min_weight > weight_map[i]){
			min_weight = weight_map[i];
		}
	}
	RGB *pixels = rgbColourFromFloat(weight_map);
	PPMImage *image = (PPMImage *)malloc(sizeof(PPMImage));
	image -> x = (char)map_side_size;
	image -> y = map_side_size;
	image -> data = pixels;

	char const * c_filename = filename.c_str();
	std::cout << "<drawPPMmap FILE: " << filename << ">" << std::endl;
	writePPM(c_filename, image);
	return 0;

	// vector< vector <float> > vector_map;
	// vector<float> current_vector;
	// for (int i = 0; i < map_size*vector_length; i++){
	// 	if (i%vector_length == 0 && i!=0){
	// 		vector_map.push_back(current_vector);
	// 		current_vector.clear();
	// 	}
	// 	current_vector.push_back(map[i]);
	// }
	// vector_map.push_back(current_vector);
	// drawMap(vector_map, filename);
}

int drawMap(vector< vector<float > > map, string filename){
	// map_side_size = sqrt((int)map.size());
	// map_vector_size = (map.at(0)).size();
	// min_weight = FLT_MAX;
	// max_weight = FLT_MIN;
	// vector<float> weight_map;
	// for (vector< vector <float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
	// 	weight_map.push_back(euclidean_distance_from_origin(*map_iter));
	// }

	// // Determin_weight max_weight and min_weight values for colour purposes
	// for (vector<float>::iterator map_iter = weight_map.begin(); map_iter != weight_map.end(); map_iter++){
	// 	if (max_weight < *map_iter){
	// 		max_weight = *map_iter;
	// 	}
	// 	else if (min_weight > *map_iter){
	// 		min_weight = *map_iter;
	// 	}
	// }
	// // string html = produceHTML(hexColourFromFloat(weight_map));
	// RGB *pixels = rgbColourFromFloat(weight_map);
	// writeToFile(html, filename);
	return 0;

}