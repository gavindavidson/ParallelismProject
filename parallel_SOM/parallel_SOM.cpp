#include <iostream>
#include <cstdlib>
#include <vector>
#include <new>
#include <sstream>
#include <ctime>
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
#define map_side_size 128
#define map_convergence_tollerance 0.02
#define vector_convergence_tollerance 0.02
#define input_size 25000
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

// float array_map[map_side_size*map_side_size*input_vector_length], array_previous_map[map_side_size*map_side_size*input_vector_length];
// float array_input[map_side_size*map_side_size*input_vector_length];
float *array_map, *array_input;
float array_previous_map[map_side_size*map_side_size*input_vector_length];

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
void print_map(float * input){
	for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i++){
		cout << input[i];
		if (i % input_vector_length == 0){
			cout << "]" << endl << "[";
		}
		else {
			cout << ",  \t";
		}
	}
}

void printArray(float *array, int size, int grouping){
	cout << "[";
	for (int i = 0; i < size; i++){
		cout << array[i] << ", ";
		if (i % grouping == 0){
			cout << "]" << endl << "	[";
		}
	}
	cout << "]";
}

float * initialiseRandomVectorsArray(int map_size, int vector_length){
	srand(time(NULL));				// Seed rand() with current time
	//float output[map_size * vector_length];
	float *output = (float *)malloc(map_size*vector_length*sizeof(float));
	for (int i = 0; i < map_size * vector_length; i++){
		output[i] = (rand()/(float)RAND_MAX) * range + min;
	}
	return output;
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

bool checkArrayConvergence(float *a, float *b, int start_index, int vector_size){
	float delta;
	for (int i = start_index; i < vector_size+start_index; i++){
		delta = a[i] * vector_convergence_tollerance;
		if (a[i] - delta >= b[i] || a[i] + delta <= b[i]){
			return false;
		}
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
	return changed_points <= map_convergence_tollerance*map.size();
}

bool arrayCovergent(){
	int changed_points = 0;
	for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i = i + input_vector_length){
		if (!checkArrayConvergence(array_map, array_previous_map, i, input_vector_length)){
			changed_points++;
		}
	}
	for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i++){
		array_previous_map[i] = array_map[i];
	}
	non_convergent_points = changed_points;
	return changed_points <= map_convergence_tollerance*map_side_size*map_side_size;
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

float array_euclidean_distance(float *a, float *b, int a_start_index, int b_start_index, int vector_size){
	float sum = 0;
	int b_index = b_start_index;
	for (int a_index = a_start_index; a_index < vector_size+a_start_index; a_index++){
		sum += pow(a[a_index] - b[b_index], 2);
		b_index++;
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
		sum += abs(*a_iter) - abs(*b_iter);
		a_iter++;
		b_iter++;
	}
	return sum;
}

float array_manhattan_distance(float *a, float *b, int a_start_index, int b_start_index, int vector_size){
	float sum = 0;
	int b_index = b_start_index;
	for (int a_index = a_start_index; a_index < vector_size+a_start_index; a_index++){
		sum += a[a_index] - b[b_index];
		b_index++;
	}
	return sum;
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

float * array_vectorMultiplyScalar(float *a, float b, int start_index, int vector_size){
	float output[vector_size];
	for (int i = start_index; i < start_index + vector_size; i++){
		output[i] = a[i]*b;
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

float * array_vectorSubtract(float *a, float *b, int start_index, int vector_size){
	float output[vector_size];
	for (int i = start_index; i < start_index + vector_size; i++){
		output[i] = a[i] - b[i];
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

float * array_vectorAdd(float *a, float *b, int start_index, int vector_size){
	float output[vector_size];
	for (int i = start_index; i < start_index + vector_size; i++){
		output[i] = a[i] + b[i];
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
	//int current_index = 0;
	int current_pos, neighbourhood_value;
	int map_size = map.size();
	for (vector< vector<float> >::iterator map_iter = map.begin(); map_iter != map.end(); map_iter++){
		current_pos = map_iter - map.end() + map_size;
		neighbourhood_value = determineSteps(current_pos, winner_index);
		if (gauss_value_list[neighbourhood_value] >= min_neighbourhood_effect){
			(*map_iter) = vectorSubtract(*map_iter, vectorMultiplyScalar(vectorSubtract(*map_iter, input_vector),gauss_value_list[neighbourhood_value]));
			/*
				current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
			*/
		}
	}
}

void array_updateWeights(int winner_index, float *input_array, int vector_size){
	int current_pos, neighbourhood_value;
	int map_size = map_side_size*map_side_size*vector_size;
	int input_offset = 0;
	for (int map_index = 0; map_index < map_size; map_index++){
		// map_index/vector_size means that neighbourhood value is the same for a 'single vector' within the whole map array
		neighbourhood_value = determineSteps(map_index/vector_size, winner_index);
		//cout << map_index << ":\told value: " << array_map[map_index];
		array_map[map_index] = array_map[map_index] - 
			((array_map[map_index] - input_array[winner_index + (map_index%vector_size)]) * gauss_value_list[neighbourhood_value]);
		//cout << "\tnew value: " << array_map[map_index] << endl;
		/*
			current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
		*/
	}
	//cout << "\n=====\n";
}

int main(){
	cout << "== Stuff to do..\t ==" << endl
			<< "\t- Make vectors into static arrays\t\t<DONE>" << endl
			<< "\t\t+ Arrays must be one dimensional" << endl
			<< "\t\t+ Fix iteration" << endl
			<< "\t- Add manhattan_distance() \t\t\t<DONE>" << endl
			<< "\t- Separate loops\t\t\t\t<IN PROGRESS>" << endl
			<< "==\t\t\t==\n" << endl;
	cout << "== Parallel SOM \t==" << endl
			<< "\t- Neighbourhood reduce iteration\t" << neighbourhood_reduce_iteration << endl
			<< "\t- Map size\t\t\t\t" << map_side_size << " x " << map_side_size << endl
			<< "\t- Map convergence tollerance\t\t" << map_convergence_tollerance << endl
			<< "\t- Vector convergence tollerance\t\t" << vector_convergence_tollerance << endl
			<< "\t- Input size\t\t\t\t" << input_size << endl
			<< "\t- Input vector length\t\t\t" << input_vector_length << endl
			<< "==\t\t\t==" << endl;

	min = 0;
	max = 10000;
	range = max - min;

	array_map = initialiseRandomVectorsArray(map_side_size*map_side_size, input_vector_length);
	for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i++){
		array_previous_map[i] = array_map[i];
	}
	array_input = initialiseRandomVectorsArray(input_size, input_vector_length);

	//drawMap(map, "map_draw/initial_map.html");
	recalculateGaussList();

	// map = initialiseRandomVectors(map_side_size*map_side_size, input_vector_length);
	// previous_map = copyMap(map);
	// input = initialiseRandomVectors(input_size, input_vector_length);
	// vector< vector <float> >::iterator map_iter, input_iter;

	//float winnerDistance, possible_winnerDistance;
	//int winner, current;
	//while (!convergent()){
	// for (i = 0; !convergent() || i == 0; i++){
	// 	// cout << "<MAP>" << endl;
	// 	// print_map(map);
	// 	// cout << "</MAP>" << endl;
	// 	time_t ctt = time(0);
	// 	cout << "Iteration: " << i << "\tNon convergent points: " << non_convergent_points << "\t" << asctime(localtime(&ctt));
	// 	for (input_iter = input.begin(); input_iter != input.end(); input_iter++){
	// 		winner = 0;
	// 		current = 0;
	// 		winnerDistance = euclidean_distance(*input_iter, *(map.begin()));
	// 		//winnerDistance = manhattan_distance(*input_iter, *(map.begin()));
			
	// 		for (map_iter = map.begin(); map_iter != map.end(); map_iter++){
	// 			possible_winnerDistance = euclidean_distance(*input_iter, *map_iter);
	// 			//possible_winnerDistance = manhattan_distance(*input_iter, *map_iter);
	// 			// cout << "IN:";
	// 			// printVector(*input_iter);
	// 			// cout << " MAP:";
	// 			// printVector(*(map.begin()));
	// 			// cout << " = " << possible_winnerDistance << endl;
	// 			if (possible_winnerDistance < winnerDistance){
	// 				winnerDistance = possible_winnerDistance;
	// 				winner = current;
	// 			}
	// 			current++;
	// 		}
	// 		//cout << input_iter - input.end() + input.size() << endl;
	// 		updateWeights(winner, *input_iter);
	// 	}
	// 	if (i%neighbourhood_reduce_iteration == 0){
	// 		gauss_value = gauss_value/2;
	// 	}
	// 	std::ostringstream convert;   // stream used for the conversion
	// 	convert << i;      
	// 	drawMap(map, "map_draw/map" + convert.str() + ".html");
	// 	cout << "<map drawn>" << endl;
	// }
	int winner, current;
	int iteration;
	int total_map_values = map_side_size*map_side_size;
	int total_input_values = input_size*input_vector_length;

	float winnerDistance, possible_winnerDistance;
	for (iteration = 0; !arrayCovergent() || iteration == 0; iteration++){
		time_t current_time = time(0);
		cout << "Iteration: " << iteration << "\tNon convergent points: " << non_convergent_points << "\t" << asctime(localtime(&current_time));
		for (int input_index = 0; input_index < total_input_values; input_index = input_index+input_vector_length){
			winner = 0;
			// float *a, float *b, int a_start_index, int b_start_index, int vector_size
			winnerDistance = array_euclidean_distance(array_map, array_input, 0, input_index, input_vector_length);
			for (int map_index = 0; map_index < total_map_values; map_index = map_index+input_vector_length){
				possible_winnerDistance = array_euclidean_distance(array_map, array_input, map_index, input_index, input_vector_length);
				if (possible_winnerDistance < winnerDistance){
					winnerDistance = possible_winnerDistance;
					winner = map_index;
				}
			}
			// int winner_index, float *input_array, int vector_size
			array_updateWeights(winner, array_input, input_vector_length);
		}
		if (iteration%neighbourhood_reduce_iteration==0 && iteration != 0){
			if (gauss_value > 1){
				gauss_value--;
				cout << "Neighbourhood reduced!\t";
			}
			else if (gauss_value >= 0.5){
				gauss_value -= 0.1;
				cout << "Neighbourhood reduced!\t";
			}
			recalculateGaussList();
		}
		//printArray(array_map, map_side_size*map_side_size*input_vector_length, input_vector_length);
		std::ostringstream convert;   // stream used for the conversion
		convert << iteration;      
		drawMap(array_map, map_side_size*map_side_size, input_vector_length, "map_draw/map" + convert.str() + ".html");
		cout << "<map drawn>" << endl;
	}
	cout << "Convergent at iteration " << iteration << "!" << endl;
	//drawMap(map, "map_draw/convergent_map.html");
	cout << "Visual representation stored at \"map_draw/convergent_map.html\"";
	//print_map(map);
	// **** 0.4 is the min gauss value.
	// gauss_value = 2;
	// for (int i = 0; gauss_value_list[0] <= 1; i++){
	// 	gauss_value = gauss_value - (0.1);
	// 	recalculateGaussList();
	// 	cout << "\ngauss_value: " << gauss_value << "\tgauss_list: " << "[" << gauss_value_list[0] << "," << gauss_value_list[1] << "," << gauss_value_list[2] << "," << gauss_value_list[3] << "]";
	// }
}