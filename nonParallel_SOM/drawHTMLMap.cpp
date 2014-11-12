#include "drawHTMLMap.h"

int map_side_size, map_vector_size;
float max_weight, min_weight;

string height = "4";
string width = "4"; //For html output

float euclidean_distance_from_origin(vector<float> a){
	float sum = 0;
	vector<float>::iterator a_iter = a.begin();
	while (a_iter != a.end()){
		sum += pow((*a_iter), 2);
		a_iter++;
	}
	//std::cout << sqrt(sum) << std::endl;
	return sqrt(sum);
}

string decimalToHex(int a){
	std::stringstream ss;
	ss<< std::hex << a; // int decimal_value
	std::string res ( ss.str() );
	if (a < 0){
		std::cout << "HIT\n";
	}
	if (a < 17){
		res = "0" + res;
	}
	return res;
}

vector<string> hexColourFromFloat(vector<float> a){
	float sextiles[6];
	float increment = (max_weight - min_weight)/6.0;
	vector<string> output;
	for (int i = 0; i < 6; i++){
		sextiles[i] = i * increment;
	}
	string current;
	int current_r_int, current_b_int, current_g_int;
	for (vector<float>::iterator a_iter = a.begin(); a_iter != a.end(); a_iter++){
		current = "#";
		if ((*a_iter) < sextiles[1]){
			current_r_int = 0;
			current_g_int = 0;
			current_b_int = ((*a_iter)/increment) * 255;
		}
		else if ((*a_iter) < sextiles[2]){
			current_r_int = 0;
			current_g_int = ((*a_iter)-sextiles[1]/increment) * 255;
			current_b_int = 255;
		}
		else if ((*a_iter) < sextiles[3]){
			current_r_int = 0;
			current_g_int = 255;
			current_b_int = 255 - (((*a_iter)-sextiles[2]/increment) * 255);
		}
		else if ((*a_iter)< sextiles[4]){
			current_r_int = ((*a_iter)-sextiles[3]/increment) * 255;
			current_g_int = 255;
			current_b_int = 0;
		}
		else{
			current_r_int = 255;
			current_g_int = 255 - (((*a_iter)-sextiles[4]/increment) * 255);
			current_b_int = 0;
		}

		std::cout << "R:" << current_r_int << " G:" << current_g_int << " B:" << current_b_int << std::endl;

		current += decimalToHex(current_r_int) + decimalToHex(current_g_int) + decimalToHex(current_b_int);
		output.push_back(current);
	}
	return output;
}

string produceHTML(vector<string> hex_colours){
	string output = "<table cellspacing=\"0\">\n<tr>";
	int current_index = 0;
	for (vector<string>::iterator colour_iter = hex_colours.begin(); colour_iter != hex_colours.end(); colour_iter++){
		if (current_index%map_side_size == 0){
			output += "\n</tr>\n<tr>";
		}
		output += string("\t<td style=\"width:") + width + "px;height:" + height + "px;background-color:" + (*colour_iter) + "\">";
		current_index++;
	}
	output += "</tr>\n</table>";
	return output;
}

int writeToFile(string contents, string filename){
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << contents;
	outfile.close();
	return 0;
}

int drawMap(vector< vector<float > > map, string filename){
	map_side_size = sqrt((int)map.size());
	map_vector_size = (map.at(0)).size();
	min_weight = FLT_MAX;
	max_weight = FLT_MIN;
	vector<float> weight_map;

	for (vector< vector <float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
		weight_map.push_back(euclidean_distance_from_origin(*map_iter));
	}

	// Determin_weighte max_weight and min_weight values for colour purposes
	for (vector<float>::iterator map_iter = weight_map.begin(); map_iter != weight_map.end(); map_iter++){
			max_weight = fmax(max_weight, *map_iter);
			min_weight = fmin(min_weight, *map_iter);
	}

	string html = produceHTML(hexColourFromFloat(weight_map));
	writeToFile(html, filename);
	return 0;

}