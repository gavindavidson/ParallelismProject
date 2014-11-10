#include <iostream>
#include <cstdlib>
#include <vector>
#include <new>
#include <cmath>

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

float gauss_value = 1.0;
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
			//current.push_back(rand()%10);
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
	float a = 1/(gauss_value*sqrt(2*pi));
	float x = steps_from_winner;
	return pow(a, (pow(-x, 2)/(2*pow(gauss_value, 2))));
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

	int x = 0;
	cout << "GAUSS VAL OF " << x << " is " << gauss(x) << endl;

	// changed_points = tollerance + 1;
	// while (!convergent()){
	// 	changed_points = 0;

	// }
}