#include <vector>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iostream>
extern "C" {
	#include "PPM.h"
}

using std::vector;
using std::string;

// Class used to draw a html representation of a map

int drawMap(vector< vector<float > >, string);
int drawMap(float *map, int map_size, int vector_length, string filename);