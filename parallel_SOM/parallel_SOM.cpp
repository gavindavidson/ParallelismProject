#include <iostream>
#include <cstdlib>
#include <vector>
#include <new>
#include <sstream>
//#include <cmath>
//#include <algorithm>
#include "../drawHTMLmap/drawHTMLMap.h"

/*
THINGS TO CONSIDER:
	-	Initial neighbourhood size
	-	When to reduce neighbourhood size
	-	Learning rate
	-	Map size
	-	Convergence tollerance
*/

// Initial neighbourhood size to be all points in map
#define neighbourhood_reduce_iteration 20
// Learning rate to be defined by a Gaussian function
#define map_side_size 64
#define tollerance 40
#define vector_convergence_tollerance 0.02
#define input_size 10000
#define input_vector_length 5

using std::vector;
using std::cout;
using std::endl;
using std::string;

float max, min, range;
float min_neighbourhood_effect = pow(10, -10);	// Minimum quotient that must be applied to a change in a point in a neighbourhood

int non_convergent_points = 0;

float gauss_value = sqrt(map_side_size);
float gauss_value_list[map_side_size];
const double pi = 3.14159265359;

vector<vector <float> > map, previous_map;
vector<vector <float> > input;

vector<vector <float> > copyMap(vector<vector <float> > map){
	vector<vector <float> > output;
	for (vector<vector <float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
		output.push_back(*map_iter);
	}
	return output;
}

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
			current.push_back(rand_val);
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

bool checkVectorConvergence(vector<float> a, vector<float> b){
	vector<float>::iterator a_iter, b_iter;
	a_iter = a.begin();
	b_iter = b.begin();
	float delta;
	while (a_iter != a.end()){
		delta = (*a_iter) * vector_convergence_tollerance;
		if ((*a_iter) - delta >= *b_iter || (*a_iter) + delta <= *b_iter){		// If b is not between a + tollernce and a - tollerence
			return false;
		}
		a_iter++;
		b_iter++;
	}
	return true;
}

bool convergent(){
	vector<vector <float> >::iterator map_iter, previous_map_iter;
	map_iter = map.begin();
	previous_map_iter = previous_map.begin();
	int changed_points = 0;
	while (map_iter != map.end()){
		if (!checkVectorConvergence(*map_iter, *previous_map_iter)){
			changed_points++;
		}
		map_iter++;
		previous_map_iter++;
	}
	previous_map = copyMap(map);
	non_convergent_points = changed_points;
	return changed_points <= tollerance;
}

/*
	Function creates a list of values where list[x] represents the value for the neighbourhood function for a point x places from the winning point
*/
void recalculateGaussList(){
	float a = 1.0/(gauss_value*sqrt(2*pi));
	//cout << "New gauss_value_list:\n[";
	for (int x = 0; x < map_side_size; x++){
		gauss_value_list[x] = a* exp(-(pow(x, 2)/(2*pow(gauss_value, 2))));
		//cout << gauss_value_list[x] << " ";
	}
	//cout << "]\n";
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

/*
	Function returns the manhattan distance between two vectors a and b
*/
float manhattan_distance(vector<float> a, vector<float> b){
	float sum = 0;
	vector<float>::iterator a_iter = a.begin();
	vector<float>::iterator b_iter = b.begin();
	while (a_iter != a.end()){
		sum += (*a_iter) - (*b_iter);
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

/*
	Function returns an int representing how close, in steps, point a is to point b
*/
int determineSteps(int a, int b){
	int a_x, a_y, b_x, b_y;
	a_x = a % map_side_size;
	a_y = a / map_side_size;
	b_x = b % map_side_size;
	b_y = b / map_side_size;

	return fmax(abs(a_x-b_x), abs(a_y-b_y));
}

/*
	Function that changes the weights of the map according to their position relative to the winning point and the input vector. This is done
	using the 
*/
void updateWeights(int winner_index, vector<float> input_vector){
	int current_index = 0;
	int current_pos, neighbourhood_value;
	int map_size = map.size();
	for (vector< vector<float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
		current_pos = map_iter - map.end() + map_size;
		neighbourhood_value = determineSteps(current_pos, winner_index);
		if (gauss_value_list[neighbourhood_value] >= min_neighbourhood_effect){
			// cout << "Original value:\t";
			// printVector(*map_iter);
			// cout << "Input vector:\t";
			// printVector(input_vector);
			// cout << "Gauss value at " << neighbourhood_value << " steps from winner" 
			// 		<< ":\t" << gauss_value_list[neighbourhood_value] << endl << "New value:\t\t";
			(*map_iter) = vectorSubtract(*map_iter, vectorMultiplyScalar(vectorSubtract(*map_iter, input_vector),gauss_value_list[neighbourhood_value]));
			// printVector(*map_iter);
			// cout << endl;
			//printVector(*map_iter);
			/*
				current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
			*/
		}
	}
}

int main(){
	cout << "== Single Threaded SOM \t==" << endl
			<< "\t- Neighbourhood reduce iteration\t" << neighbourhood_reduce_iteration << endl
			<< "\t- Map size\t\t\t\t" << map_side_size << " x " << map_side_size << endl
			<< "\t- Convergence tollerance\t\t" << tollerance << endl
			<< "\t- Vector convergence tollerance\t\t" << vector_convergence_tollerance << endl
			<< "\t- Input size\t\t\t\t" << input_size << endl
			<< "\t- Input vector length\t\t\t" << input_vector_length << endl
			<< "==\t\t\t==" << endl;

	min = 0;
	max = 10000;
	range = max - min;
	map = initialiseRandomVectors(map_side_size*map_side_size, input_vector_length);
	previous_map = copyMap(map);

	drawMap(map, "map_draw/initial_map.html");
	recalculateGaussList();

	input = initialiseRandomVectors(input_size, input_vector_length);

	vector< vector <float> >::iterator map_iter, input_iter;

	float winnerDistance, possible_winnerDistance;
	int winner, current;
	//while (!convergent()){
	int i;
	for (i = 0; !convergent() || i == 0; i++){
		// cout << "<MAP>" << endl;
		// print_map(map);
		// cout << "</MAP>" << endl;
		cout << "Iteration: " << i << "\tNon convergent points: " << non_convergent_points << endl;
		for (input_iter = input.begin(); input_iter != input.end(); input_iter++){
			winner = 0;
			current = 0;
			//winnerDistance = euclidean_distance(*input_iter, *(map.begin()));
			winnerDistance = manhattan_distance(*input_iter, *(map.begin()));
			
			for (map_iter = map.begin(); map_iter != map.end(); map_iter++){
				possible_winnerDistance = euclidean_distance(*input_iter, *map_iter);
				if (possible_winnerDistance < winnerDistance){
					winnerDistance = possible_winnerDistance;
					winner = current;
				}
				current++;
			}
			//cout << input_iter - input.end() + input.size() << endl;
			updateWeights(winner, *input_iter);
		}
		if (i%neighbourhood_reduce_iteration == 0){
			gauss_value = gauss_value/2;
			std::ostringstream convert;   // stream used for the conversion
			convert << i;      
			drawMap(map, "map_draw/map" + convert.str() + ".html");
			cout << "<map drawn>" << endl;
		}
	}
	cout << "Convergent at iteration " << i << "!" << endl;
	drawMap(map, "map_draw/convergent_map.html");
	cout << "Visual representation stored at \"map_draw/convergent_map.html\"";
	//print_map(map);
}