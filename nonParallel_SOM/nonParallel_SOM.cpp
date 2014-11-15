#include <iostream>
#include <cstdlib>
#include <vector>
#include <new>
#include <sstream>
//#include <cmath>
//#include <algorithm>
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
#define neighbourhood_reduce_iteration 10
// Learning rate to be defined by a Gaussian function
#define map_side_size 4
#define tollerance 40
#define input_size 1024
#define input_vector_length 5

using std::vector;
using std::cout;
using std::endl;
using std::string;

float max, min, range;
int changed_points;
float min_neighbourhood_effect = 0.00001;	// Minimum quotient that must be applied to a change in a point in a neighbourhood


//float gauss_value = sqrt(map_side_size);
float gauss_value = 1;
float gauss_value_list[map_side_size];
const double pi = 3.14159265359;

vector<vector <float> > map;
vector<vector <float> > input;

/*
	Function creates a vector of random float vectors, making use of psudo-random number generation based on current time
*/
vector<vector <float> > initialiseRandomVectors(int map_size, int vector_length){
	srand(time(NULL));				// Seed rand() with current time
	vector<vector <float> > output;
	vector<float> current;
	for (int i = 0; i < map_size; i++){
		current.clear();	
		for (int k = 0; k < vector_length; k++){
			float rand_val = (rand()/(float)RAND_MAX) * range + min;
			//float rand_val = rand()/(float)RAND_MAX;
			current.push_back(rand_val);
			//cout << rand_val << "\t";
		}
		output.push_back(current);
	}
	return output;
}

/*
	Function outputs a string representation to cout of the map
*/
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

/*
	Function creates a list of values where list[x] represents the value for the neighbourhood function for a point x places from the winning point
*/
void recalculateGaussList(){
	float a = 1.0/(gauss_value*sqrt(2*pi));
	for (int x = 0; x < map_side_size; x++){
		gauss_value_list[x] = a* exp(-(pow(x, 2)/(2*pow(gauss_value, 2))));
	}
}

/*
	Function returns the eucluidean distance between two vectors a and b
*/
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

vector<float> vectorMultiplyScalar(vector<float> a, float b){
	vector<float> output;
	vector<float>::iterator a_iter = a.begin();
	while (a_iter != a.end()){
		output.push_back((*a_iter) * b);
		//cout << (*a_iter) << " x " <<  b << " = " << (*a_iter) * b << endl;
		a_iter++;
	}
	return output;
}

vector<float> vectorSubtract(vector<float> a, vector<float> b){
	vector<float> output;
	vector<float>::iterator a_iter = a.begin();
	vector<float>::iterator b_iter = b.begin();
	while (a_iter != a.end()){
		output.push_back((*a_iter) - (*b_iter));
		a_iter++;
		b_iter++;
	}
	return output;
}

vector<float> vectorAdd(vector<float> a, vector<float> b){
	vector<float> output;
	vector<float>::iterator a_iter = a.begin();
	vector<float>::iterator b_iter = b.begin();
	while (a_iter != a.end()){
		output.push_back((*a_iter) + (*b_iter));
		a_iter++;
		b_iter++;
	}
	return output;
}

void printVector(vector<float> a){
	cout << "[";
	for (vector<float>::iterator a_iter = a.begin(); a_iter != a.end(); a_iter++){
		cout << (*a_iter) << ",";
	}
	cout << "]" << endl;
}

void updateWeights(int winner_index, vector<float> input_vector){
	int current_index = 0;
	int current_pos;
	int map_size = map.size();
	for (vector< vector<float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
		current_pos = map_iter - map.end() + map_size;
		current_pos = 0;
		//if (gauss_value_list[current_pos] >= min_neighbourhood_effect){
			(*map_iter) = vectorSubtract(*map_iter, vectorMultiplyScalar(vectorSubtract(*map_iter, input_vector),gauss_value_list[current_pos]));
			//printVector(*map_iter);
			/*
				current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
			*/
		//}
	}
}

/*
	Function returns an int representing how close, in steps, point a is to point b
*/
int determineSteps(int a, int b){
	int a_x, a_y, b_x, b_y;
	a_x = a % map_side_size;
	a_y = a / map_side_size;
	b_x = a % map_side_size;
	b_y = a / map_side_size;

	return fmax(abs(a_x-b_x), abs(a_y-b_y));
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

	// vector<float> test_vec, out_vec;
	// test_vec.push_back(1);
	// test_vec.push_back(2);
	// test_vec.push_back(3);
	// test_vec.push_back(4);
	// test_vec.push_back(5);

	// out_vec = vectorSubtract(test_vec, test_vec);
	// for (vector<float>::iterator out_vec_iter = out_vec.begin(); out_vec_iter != out_vec.end(); out_vec_iter++){
	// 	cout << *out_vec_iter << endl;
	// }

	min = 0;
	max = 10000;
	range = max - min;
	map = initialiseRandomVectors(map_side_size*map_side_size, input_vector_length);
	drawMap(map, "map_draw/initial_map.html");
	recalculateGaussList();
	// cout << "Gauss List: [";
	// for (int i = 0; i < map_side_size; i++){
	// 	cout << gauss_value_list[i] << ",\t";
	// }
	// for (vector< vector <float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
	// 	for (vector<float>::iterator vector_iter = (*map_iter).begin(); vector_iter != (*map_iter).end(); vector_iter++){
	// 		cout << (*vector_iter) << endl;
	// 	}
	// }
	input = initialiseRandomVectors(input_size, input_vector_length);

	vector< vector <float> >::iterator map_iter, input_iter;
	// vector< <float> >::iterator input_iter;

	changed_points = tollerance + 1;
	float winnerDistance, possible_winnerDistance;
	int winner, current;
	//while (!convergent()){
	for (int i = 0; i < 1000; i++){
		print_map(map);
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
			cout << endl;
			printVector(map.at(winner));
			cout << "\tindex:\t" << winner << endl;
			updateWeights(winner, *input_iter);
		}
		cout << "Index: " << i << endl;
		cout << endl;
		if (i%15 == 0){
			gauss_value = gauss_value/2;
			std::ostringstream convert;   // stream used for the conversion
			convert << i;      
			//drawMap(map, "map_draw/map" + convert.str() + ".html");
			//cout << "New Map drawn" << endl;
			//print_map(map);
		}
	}
}