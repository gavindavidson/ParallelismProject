#include <iostream>
#include <cstdlib>
#include <vector>
#include <new>
#include <cmath>
#include <algorithm>
#include "drawHTMLMap.h"

/*
THINGS TO CONSIDER:
	-	Initial neighbourhood size
	-	When to reduce neighbourhood size
	-	Learning rate
	-	Map size
	-	Convergence tollerance
*/

// Initial neighbourhood size to be all points in map
#define neighbourhood_reduce_iteration 100
// Learning rate to be defined by a Gaussian function
#define map_side_size 64
#define tollerance 40
#define input_size 16384
#define input_vector_length 5

using std::vector;
using std::cout;
using std::endl;
using std::string;

float max, min, range;
int changed_points;

float gauss_value = sqrt(map_side_size);
const double pi = 3.14159265359;

vector<vector <float> > map;
vector<vector <float> > input;

vector<vector <float> > initialiseRandomVectors(int map_size, int vector_length){
	vector<vector <float> > output;
	vector<float> current;
	for (int i = 0; i < map_size; i++){
		current.clear();	
		for (int k = 0; k < vector_length; k++){
			float rand_val = rand()/(RAND_MAX/range) + min;
			current.push_back(rand_val);
		}
		output.push_back(current);
	}
	return output;
}

void print_map(vector<vector <float> > input){
	vector<vector <float> >::iterator map_iterator;
	vector<float>::iterator vector_iterator;

	for (map_iterator = input.begin(); map_iterator != input.end(); map_iterator++){
		cout << "[ ";
		for (vector_iterator = (*map_iterator).begin(); vector_iterator != (*map_iterator).end(); vector_iterator++){
			cout << (*vector_iterator) << "\t";
		}
		cout << "]" << endl;
	}
}

bool convergent(){
	return changed_points < tollerance;
}

float gauss(int steps_from_winner){
	float a = 1.0/(gauss_value*sqrt(2*pi));
	float x = steps_from_winner;
	return a* exp(-(pow(x, 2)/(2*pow(gauss_value, 2))));
}

float euclidean_distance(vector<float> a, vector<float> b){
	float sum = 0;
	vector<float>::iterator a_iter = a.begin();
	vector<float>::iterator b_iter = b.begin();
	while (a_iter != a.end()){
		sum += pow((*a_iter) - (*b_iter), 2);
		a_iter++;
		b_iter++;
	}
	return sqrt(sum);
}

void updateWeights(int winner_index){
	// int current_index 0;
	// for (vector<float>::iterator map_iter = map.begin(); map_iter != map_iter.end(); map_iter++){

	// }
}

int determineSteps(int a, int b){
	// int a_x, a_y, b_x, b_y;
	// a_x = a % map_side_size;
	// a_y = a / map_side_size;
	// b_x = a % map_side_size;
	// b_y = a / map_side_size;

	// return max(abs(a_x-b_x), abs(a_y-b_y));
	return 0;
}


int main(){
	cout << "== Single Threaded SOM \t==" << endl
			<< "\t- Neighbourhood reduce iteration\t" << neighbourhood_reduce_iteration << endl
			<< "\t- Map size\t\t\t\t" << map_side_size << " x " << map_side_size << endl
			<< "\t- Convergence tollerance\t\t" << tollerance << endl
			<< "\t- Input size\t\t\t\t" << input_size << endl
			<< "\t- Input vector length\t\t\t" << input_vector_length << endl
			<< "==\t\t\t==" << endl;

	min = 0;
	max = 10000;
	range = max - min;
	map = initialiseRandomVectors(map_side_size*map_side_size, input_vector_length);
	input = initialiseRandomVectors(input_size, input_vector_length);

	vector< vector <float> >::iterator map_iter, input_iter;
	//vector< <float> >::iterator input_iter;

	changed_points = tollerance + 1;
	float winnerDistance, possible_winnerDistance;
	int winner, current;
	while (!convergent()){
		for (input_iter = input.begin(); input_iter != input.end(); input_iter++){
			winner = 0;
			current = 0;
			winnerDistance = euclidean_distance(*input_iter, *(map.begin()));
			for (map_iter = map.begin(); map_iter != map.end(); map_iter++){
				possible_winnerDistance = euclidean_distance(*input_iter, *map_iter);
				if (possible_winnerDistance < winnerDistance){
					winnerDistance = possible_winnerDistance;
					winner = current;
				}
				current++;
			}
			updateWeights(winner);
		}
	}
}