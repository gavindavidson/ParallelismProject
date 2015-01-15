#include <iostream>
#include <cstdlib>
#include <vector>
#include <new>
#include <sstream>
#include <ctime>
#include <string>
//#include "../drawHTMLmap/drawHTMLMap.h"
#include "../ppm/drawPPMmap.h"


// PRE OPENCL TESTING
#include "manhattan_distance.h"
#include "update_weight.h"
// =================

/*
THINGS TO CONSIDER:
	-	Initial neighbourhood size
	-	When to reduce neighbourhood size
	-	Learning rate
	-	Map size
	-	Convergence tollerance
*/

// Initial neighbourhood size to be all points in map
#define cycle_length 20
// Learning rate to be defined by a Gaussian function
#define map_side_size 16
#define trials 3
#define map_convergence_tollerance 0.00
#define vector_convergence_tollerance 0.000001

#define input_size 2000
#define input_vector_length 3
#define input_data_clusters 5

using std::vector;
using std::cout;
using std::endl;
using std::string;

float max, min, range;
float min_neighbourhood_effect = pow(10, -10);	// Minimum quotient that must be applied to a change in a point in a neighbourhood

int non_convergent_points = 0;

float gauss_value = sqrt(map_side_size)/10;
float gauss_value_list[map_side_size];
const double pi = 3.14159265359;

float *map, *input, *previous_map, *distance_map, *best_map;
float best_quantisation_error;

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
		if (i % grouping == 0){
			cout << "]" << endl << i/grouping << "[";
		}
		cout << array[i];
		if (i % grouping != grouping - 1){
			cout  << ", ";
		}
	}
	cout << "]";
}

int writeToFile(float *data, int size, string filename){
	string contents = "";
	for (int i = 0; i < size*input_vector_length; i++){
		if (i%input_vector_length == 0){
			contents += "\n";
		}
		std::ostringstream convert;
		convert << data[i]; 
		contents += convert.str() + "\t";
	}
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << contents;
	outfile.close();
	return 0;
}


/*
	Function produces a random dataset within certain predifined ranges
*/
float * initialiseRandomArray(int array_size, int vector_length){
	srand(time(NULL));				// Seed rand() with current time
	cout << "<Producing random array>" << endl;
	//float output[map_size * vector_length];
	float *output = (float *)malloc(array_size*vector_length*sizeof(float));
	for (int i = 0; i < (array_size * vector_length); i++){
		output[i] = (rand()/(float)RAND_MAX) * range + min;
	}
	return output;
}

/*
	Function produces a random dataset that is purposefully clustered for testing
*/
float * initialiseClusteredArray(int array_size, int vector_length, int clusters){
	srand(time(NULL));
	cout << "<Producing " << clusters << " cluster array >" << endl;
	float *output = (float *)malloc(array_size*vector_length*sizeof(float));
	float centre;
	float max_variance = max/20;	// Clusters are to be within 5% of the max value of the centre point
	for (int outer = 0; outer < clusters; outer++){
		centre = (rand()/(float)RAND_MAX) * range + min;
		for (int inner = 0; inner < (array_size * vector_length)/clusters; inner++){
			output[inner + (outer*(array_size * vector_length)/clusters)] = (rand()/(float)RAND_MAX) * max_variance + centre;
			//cout << inner + (outer*(array_size * vector_length)/clusters) << " ";
		}
	}
	for (int i = 0; i < (array_size * vector_length)%clusters; i++){
		output[(array_size * vector_length) - 1 - i] = (rand()/(float)RAND_MAX) * range + min;
	}
	return output;
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
	int changed_points = 0;
	for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i = i + input_vector_length){
		if (!checkArrayConvergence(map, previous_map, i, input_vector_length)){
			changed_points++;
		}
	}
	for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i++){
		previous_map[i] = map[i];
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

		// if (x == 0){
		// 	gauss_value_list[x] = 1;
		// }
		// else {
		// 	gauss_value_list[x] = 0;
		// }
		//cout << gauss_value_list[x] << " ";
	}
	//cout << "]\n";
}

/*
	Prototype function that moves each element of the gauss list one place to the left, replacing unknown values with zeros.
*/
void shuntGaussList(){
	float temp_value;
	for (int i = 1; i < map_side_size; i++){
		temp_value = gauss_value_list[i];
		gauss_value_list[i-1] = temp_value;
	}
	gauss_value_list[map_side_size-1] = 0;
}

/*
	Function returns the eucluidean distance between two vectors a and b
*/
float euclidean_distance(float *a, float *b, int a_start_index, int b_start_index, int vector_size){
	float sum = 0;
	int b_index = b_start_index;
	for (int a_index = a_start_index; a_index < vector_size + a_start_index; a_index++){
		sum += pow(a[a_index] - b[b_index], 2);
		b_index++;
	}
	return sqrt(sum);
}

/*
	Function returns the manhattan distance between two vectors a and b
*/
// float manhattan_distance(float *a, float *b, int a_start_index, int b_start_index, int vector_size){
// 	float sum = 0;
// 	int b_index = b_start_index;
// 	for (int a_index = a_start_index; a_index < vector_size+a_start_index; a_index++){
// 		sum += abs(a[a_index] - b[b_index]);
// 		b_index++;
// 	}
// 	return sum;
// }

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
	int a_x, a_y, b_x, b_y, output;
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
void update_weights(int winner_index, int input_start_index, int vector_size){
	int current_pos, neighbourhood_value;
	int map_size = map_side_size*map_side_size*vector_size;
	// for (int map_index = 0; map_index < map_size; map_index++){
	// 	// map_index/vector_size means that neighbourhood value is the same for a 'single vector' within the whole map array
	// 	neighbourhood_value = determineSteps(map_index/vector_size, winner_index);
	// 	//cout << endl << map_index << ":\told value: " << map[map_index];
	// 	map[map_index] = map[map_index] - 
	// 			((map[map_index] - input_array[input_start_index + (map_index%vector_size)]) * gauss_value_list[neighbourhood_value]);

	for (int current_position = 0; current_position < map_side_size*map_side_size; current_position++){
		/*
		void update_weight(
		float *map,
		float *input_array,
		float *gauss_value_list,
		int winner_index,
		int input_start_index,
		int vector_length,
		int map_side_size,
		int current_id		// ONLY FOR NON OPENCL VERSION
		);
		*/
		update_weight(map, input, gauss_value_list, winner_index, input_start_index, input_vector_length, map_side_size, current_position);

		//cout << "\tnew value: " << map[map_index] << "\tN_value: " << gauss_value_list[neighbourhood_value] << "\tIn: " << input_array[input_start_index] << "\tW_index: " << winner_index << endl;
		/*
			current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
		*/
	}
	//cout << "\n\n=====\n\n";
}

int findWinner(int input_index){
	int winner = 0;
	int total_map_values = map_side_size*map_side_size*input_vector_length;
	float winnerDistance = FLT_MAX;
	//winnerDistance = euclidean_distance(map, input, 0, input_index, input_vector_length);
	//winnerDistance = manhattan_distance(map, input, 0, input_index, input_vector_length);
	//for (int map_index = 0; map_index < total_map_values; map_index = map_index+input_vector_length){ 
		// distance_map[map_index/input_vector_length] = euclidean_distance(map, input, map_index, input_index, input_vector_length);
		//distance_map[map_index/input_vector_length] = manhattan_distance(map, input, map_index, input_index, input_vector_length);

	// -- OPENCL KERNEL	--
	for (int current_iter = 0; current_iter < map_side_size*map_side_size; current_iter++){
		manhattan_distance(&input[input_index], map, distance_map, input_vector_length, current_iter);
	}
	// --				--

	for (int distance_index = 0; distance_index < map_side_size*map_side_size; distance_index++){
		if (distance_map[distance_index] < winnerDistance){
			winnerDistance = distance_map[distance_index];
			winner = distance_index;
		}
	}
	return winner;
}
// float euclidean_distance(float *a, float *b, int a_start_index, int b_start_index, int vector_size){
float quantisationError(int input_index){
	int winner = findWinner(input_index);
	float local_average_error = 0;
	for (int i = 0; i < input_vector_length; i++){
		local_average_error += abs(input[input_index] - map[winner*input_vector_length]);
	}
	local_average_error = local_average_error/input_vector_length;
	return local_average_error;

	// float total_error_percentage = 0;
	// float total_error = 0.0;
	// float percentage, difference;
	// difference = euclidean_distance(input, map, input_index, winner, input_vector_length);
	// return difference;


	// for (int i = 0; i < input_vector_length; i++){
	// 	//percentage = (map[winner + i] - input[input_index + i])/input[input_index + i];
	// 	difference = (map[winner + i] - input[input_index + i]);
	// 	// if (percentage >= 0){
	// 	// 	total_error_percentage += percentage;
	// 	// }
	// 	// else {
	// 	// 	total_error_percentage -= percentage;
	// 	// }
	// 	if (difference < 0){
	// 		difference = difference*-1;
	// 	}
	// 	total_error += sqrt(difference);
	// }
	//return total_error_percentage/input_vector_length;
	// return total_error;
}

void drawProgessBar(int current, int max){
	int percentage = ((float)current/max)*100;
	cout << "\r";
	cout << percentage << "%\t|";
	for (int i = 0; i < percentage/2; i++){
		cout << "=";
	}
	for (int i = percentage/2; i < 50; i++){
		cout << " ";
	}
	cout << "|" << std::flush;
}

int main(){
	cout << "== Stuff to do..\t ==" << endl
			<< "\t- Make vectors into static arrays\t\t<DONE>" << endl
			<< "\t\t+ Arrays must be one dimensional" << endl
			<< "\t\t+ Fix iteration" << endl
			<< "\t- Add manhattan_distance() \t\t\t<DONE>" << endl
			<< "\t- Separate loops\t\t\t\t<DONE>" << endl
			<< "\t- Set up optimal map finding\t\t\t<DONE>" << endl
			<< "\t\t+ Set up quantisation error checker" << endl
			<< "\t\t+ Set up repeated map building routine" << endl
			<< "\t- Tune gaussian curve\t\t\t\t<IN PROGRESS>" << endl
			<< "\t- Set up openCL version\t\t\t\t" << endl
			<< "\t\t+ Put functions into C code" << endl
			<< "\t\t+ Put functions into separate files" << endl
			<< "\t\t+ Add openCL stuff" << endl
			<< "==\t\t\t==\n" << endl;
	cout << "== Parallel SOM \t==" << endl
			<< "\t- Cycle length\t\t\t\t" << cycle_length << endl
			<< "\t- Map size\t\t\t\t" << map_side_size << " x " << map_side_size << endl
			// << "\t- Map convergence tollerance\t\t" << map_convergence_tollerance << endl
			// << "\t- Vector convergence tollerance\t\t" << vector_convergence_tollerance << endl
			<< "\t- Input size\t\t\t\t" << input_size << endl
			<< "\t- Input vector length\t\t\t" << input_vector_length << endl
			<< "\t- Trials\t\t\t\t" << trials << endl
			<< "(\t- gauss_value\t\t\t\t" << gauss_value << ")" << endl
 			<< "==\t\t\t==" << endl;

	min = 0;
	max = 10000;
	range = max - min;

	previous_map = (float *)malloc(sizeof(float)*map_side_size*map_side_size*input_vector_length);
	distance_map = (float *)malloc(sizeof(float)*map_side_size*map_side_size);

	// input = initialiseRandomArray(input_size, input_vector_length);
	// writeToFile(input, input_size, "input.dat");
	input = initialiseClusteredArray(input_size, input_vector_length, input_data_clusters);
	writeToFile(input, input_size, "input_clustered.dat");

	//drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/initial_map.html");

	int current;
	int *winner = (int *)malloc(sizeof(int));
	int iteration;
	int total_map_values = map_side_size*map_side_size*input_vector_length;
	int total_input_values = input_size*input_vector_length;
	// map = initialiseRandomArray(map_side_size*map_side_size, input_vector_length);
	//for (iteration = 0; !convergent() || iteration == 0; iteration++){
	for (int current_trial = 0; current_trial < trials; current_trial++){
		map = initialiseRandomArray(map_side_size*map_side_size, input_vector_length);
		drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/initial_map.ppm");
		for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i++){
			previous_map[i] = map[i];
		}
		recalculateGaussList();
		writeToFile(gauss_value_list, map_side_size, "learning_rates.dat");
		time_t current_time = time(0);
		cout << "TRIAL " << current_trial << ": started at " << asctime(localtime(&current_time));
		// cout << "TRIAL " << current_trial << endl;
		for (iteration = 0; iteration < cycle_length*map_side_size; iteration++){
			drawProgessBar(iteration, cycle_length*map_side_size);
			//current_time = time(0);
			//cout << "Iteration: " << iteration << "\tNon convergent points: " << non_convergent_points << "\t" << asctime(localtime(&current_time));
			//cout << "Iteration: " << iteration << "\t" << asctime(localtime(&current_time));
			for (int input_index = 0; input_index < total_input_values; input_index = input_index+input_vector_length){
				*winner = findWinner(input_index);
				// winner = 1;
				update_weights(*winner, input_index, input_vector_length);
			}
			if (iteration%cycle_length==0 && iteration != 0){
				// if (gauss_value > 1){
				// 	gauss_value--;
				// 	cout << "Neighbourhood reduced\t";
				// 	recalculateGaussList();
				// }
				// else if (gauss_value >= 0.5){
				// 	gauss_value -= 0.1;
				// 	cout << "Neighbourhood reduced\t";
				// 	recalculateGaussList();
				// }
				if (gauss_value_list[1] != 0){
					//cout << "Neighbourhood reduced\t" << endl;
					shuntGaussList();
				}
				// std::ostringstream convert;   // stream used for the conversion
				// convert << iteration;      
				// drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/map" + convert.str() + ".html");
				// cout << "<map drawn>" << endl;
			}
			// std::ostringstream convert;   // stream used for the conversion
			// convert << iteration;      
			// drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/map" + convert.str() + ".html");
			// cout << "<map drawn>" << endl;
			//printArray(map, map_side_size*map_side_size*input_vector_length, input_vector_length);

		}
		drawProgessBar(cycle_length*map_side_size, cycle_length*map_side_size);
		//cout << "Convergent at iteration " << iteration << "!" << endl;
		//cout << "Completeion at iteration " << iteration << "!" << endl;
		// drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/convergent_map.html");
		float total_quantisation_error = 0;
		for (int input_index = 0; input_index < total_input_values; input_index = input_index+input_vector_length){
			total_quantisation_error += quantisationError(input_index);
		}
		cout << endl << "Average Quantisation Error: " << total_quantisation_error/input_size << endl;
		std::ostringstream convert;   // stream used for the conversion
		convert << current_trial;
		drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/map_trial_" + convert.str() + ".ppm");
		writeToFile(map, map_side_size*map_side_size, "map_"+convert.str() + ".dat");
		if (current_trial == 0){
			best_quantisation_error = total_quantisation_error;
			best_map = map;
		}
		else if (total_quantisation_error < best_quantisation_error){
			best_quantisation_error = total_quantisation_error;
			free(best_map);
			best_map = map;
		}
	}
	// cout << "Visual representation stored at \"map_draw/convergent_map.html\"" << endl;
	cout << "Process complete\nBest quantisation error: " << best_quantisation_error/input_size << endl;
	drawMap(best_map, map_side_size*map_side_size, input_vector_length, "map_draw/best_map.ppm");
	writeToFile(best_map, map_side_size*map_side_size, "map.dat");
	cout << "Visual representation stored at \"map_draw/best_map.ppm\"" << endl;
	time_t current_time = time(0);
	cout << "FINISHED at " << asctime(localtime(&current_time));
}