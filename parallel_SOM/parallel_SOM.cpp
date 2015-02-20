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
// #include "manhattan_distance.h"
// #include "update_weight.h"
// =================
#include "cl.hpp"
#include "util.hpp"

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
// #define map_side_size 32
#define trials 3
#define map_convergence_tollerance 0.00
#define vector_convergence_tollerance 0.000001

// #define input_size 5120
// #define input_vector_length 3
#define input_data_clusters 5

using std::cout;
using std::endl;
using std::string;

float max, min, range;
float min_neighbourhood_effect = pow(10, -10);	// Minimum quotient that must be applied to a change in a point in a neighbourhood

int input_vector_length = 0;
int input_size = 0;
int map_side_size = 0;

// float gauss_value = sqrt(map_side_size)/10;
float gauss_value = 7.0/10;
float *gauss_value_list;
const double pi = 3.14159265359;

float *map, *input, *previous_map, *distance_map, *best_map, *winner_distance_array;
int *winner_index_array;
float best_quantisation_error;

std::clock_t local_start_time, global_start_time;
int update_weight_time, manhattan_distance_time, min_distance_time, min_distance_read_time;

// OPENCL
int compute_units;
cl_int err;
cl::Buffer map_buffer, distance_map_buffer, input_buffer, gauss_value_list_buffer, winner_index_buffer, output_buffer, winner_index_array_buffer, winner_distance_array_buffer;
//cl::Buffer subject_vector_buffer;
cl::Context CPU_context;

// cl::Program manhattan_distance_prog;
cl::Kernel manhattan_distance_kernel, update_weight_kernel, min_distance_kernel;
cl::CommandQueue command_queue;

inline void
    checkErr(cl_int err, const char * name)
    {
    if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name
    << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
    }
}
// ---

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

float* readInputFromFile(string filename){
	cout << "<Reading input file>" << endl;
	string line;
	std::ifstream map_file;
	map_file.open(filename.c_str());
	int count = 0;
	if (!map_file.is_open()){
		cout << "File \"" << filename << "\" could not be opened" << endl;
		exit(0);
	}
	map_file >> input_size;
	map_file >> input_vector_length;
	float *output = (float *)malloc(input_size*input_vector_length*sizeof(float));
	for (int i = 0; i < input_size * input_vector_length; i++){
		map_file >> output[i];
	}
	map_file.close();
	return output;
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
	std::cout << "<Producing random array>" << std::endl;
	//float output[map_size * vector_length];
	float *output = (float *)malloc(array_size*vector_length*sizeof(float));
	for (int i = 0; i < (array_size * vector_length); i++){
		output[i] = (rand()/(float)RAND_MAX) * range + min;
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

/*
	Function creates a list of values where list[x] represents the value for the neighbourhood function for a point x places from the winning point
*/
void calculateGaussList(){
	float a = 1.0/(gauss_value*sqrt(2*pi));
	//cout << "New gauss_value_list:\n[";
	for (int x = 0; x < map_side_size; x++){
		gauss_value_list[x] = a* exp(-(pow(x/5.0, 2)/(2*pow(gauss_value, 2))));
		
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
// int determineSteps(int a, int b){
// 	int a_x, a_y, b_x, b_y, output;
// 	a_x = a % map_side_size;
// 	a_y = a / map_side_size;
// 	b_x = b % map_side_size;
// 	b_y = b / map_side_size;

// 	return fmax(abs(a_x-b_x), abs(a_y-b_y));
// }

/*
	Function that changes the weights of the map according to their position relative to the winning point and the input vector. This is done
	using the 
*/
void update_weights(int input_start_index, int vector_size){
	int current_pos, neighbourhood_value;
	int map_size = map_side_size*map_side_size*vector_size;
	// for (int map_index = 0; map_index < map_size; map_index++){
	// 	// map_index/vector_size means that neighbourhood value is the same for a 'single vector' within the whole map array
	// 	neighbourhood_value = determineSteps(map_index/vector_size, winner_index);
	// 	//cout << endl << map_index << ":\told value: " << map[map_index];
	// 	map[map_index] = map[map_index] - 
	// 			((map[map_index] - input_array[input_start_index + (map_index%vector_size)]) * gauss_value_list[neighbourhood_value]);

	// <OPENCL>
	// need winner index and input start index
	// update_weight_kernel.setArg(3, winner_index);
	// checkErr(err, "update_weight_kernel: kernel(3)");

	update_weight_kernel.setArg(4, input_start_index);
	checkErr(err, "update_weight_kernel: kernel(4)");

	// start = std::clock();
 //     // your test
 //     std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
	local_start_time = std::clock();
	cl::Event end_event;
	err = command_queue.enqueueNDRangeKernel(update_weight_kernel, cl::NullRange, cl::NDRange(map_side_size*map_side_size), cl::NullRange	, NULL, &end_event);
	checkErr(err, "update_weight_kernel: enqueueNDRangeKernel()");

	end_event.wait();
	update_weight_time += (std::clock() - local_start_time);

	// int *array = (int *)malloc(sizeof(int)*map_side_size*map_side_size*map_side_size);
	// err = command_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0,
	// 	map_side_size*map_side_size*sizeof(int), array);
	// checkErr(err, "winner_index_buffer: enqueueReadBuffer()");
	// //cout << "array[0] = " << array[0];
	// for (int i = 0; i < map_side_size*map_side_size; i++){
	// 	if (array[i] != array[0]){
	// 		//cout << "FAIL: " << array[i] << endl;
	// 		//break;
	// 		cout << "\tIndex: " << i << " = " << array[i] << endl;
	// 	}
	// }
	// </OPENCL>

	// for (int current_position = 0; current_position < map_side_size*map_side_size; current_position++){
	// 	/*
	// 	void update_weight(
	// 	float *map,
	// 	float *input_array,
	// 	float *gauss_value_list,
	// 	int winner_index,
	// 	int input_start_index,
	// 	int vector_length,
	// 	int map_side_size,
	// 	int current_id		// ONLY FOR NON OPENCL VERSION
	// 	);
	// 	*/
	// 	update_weight(map, input, gauss_value_list, winner_index, input_start_index, input_vector_length, map_side_size, current_position);

	// 	//cout << "\tnew value: " << map[map_index] << "\tN_value: " << gauss_value_list[neighbourhood_value] << "\tIn: " << input_array[input_start_index] << "\tW_index: " << winner_index << endl;
	// 	/*
	// 		current_pos_vector =  current_pos_vector - ((current_pos_vector - input_vector) * neighbourhood_function)
	// 	*/
	// }
	//cout << "\n\n=====\n\n";
}

void findWinner(int input_index){
	int winner = 0;
	int total_map_values = map_side_size*map_side_size*input_vector_length;
	float winnerDistance = FLT_MAX;
	//winnerDistance = euclidean_distance(map, input, 0, input_index, input_vector_length);
	//winnerDistance = manhattan_distance(map, input, 0, input_index, input_vector_length);
	//for (int map_index = 0; map_index < total_map_values; map_index = map_index+input_vector_length){ 
		// distance_map[map_index/input_vector_length] = euclidean_distance(map, input, map_index, input_index, input_vector_length);
		//distance_map[map_index/input_vector_length] = manhattan_distance(map, input, map_index, input_index, input_vector_length);

	// <PRE OPENCL KERNEL>
	// for (int current_iter = 0; current_iter < map_side_size*map_side_size; current_iter++){
	// 	manhattan_distance(&input[input_index], map, distance_map, input_vector_length, current_iter);
	// }
	// </PRE OPENCL KERNEL>

	// <OPENCL>
	// subject_vector_buffer = cl::Buffer(CPU_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
	// 	(size_t)input_vector_length, &input[input_index], &err);
	// checkErr(err, "subject_vector_buffer");

	// err = command_queue.enqueueWriteBuffer(subject_vector_buffer, CL_TRUE, 0, input_vector_length, &input[input_index]);
	// checkErr(err, "subject_vector_buffer: enqueueWriteBuffer()");

	// manhattan_distance_kernel.setArg(0, subject_vector_buffer);
	// // checkErr(err, "kernel(0)");
	manhattan_distance_kernel.setArg(4, input_index);
	checkErr(err, "manhattan_distance_kernel: kernel(4)");

	local_start_time = std::clock();

	cl::Event end_event;
	err = command_queue.enqueueNDRangeKernel(manhattan_distance_kernel, cl::NullRange, cl::NDRange(map_side_size*map_side_size), cl::NullRange, NULL, &end_event);
	checkErr(err, "manhattan_distance_kernel: enqueueNDRangeKernel()");

	end_event.wait();

	manhattan_distance_time += (std::clock() - local_start_time);
	local_start_time = std::clock();

	//err = command_queue.enqueueNDRangeKernel(min_distance_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &end_event);
	err = command_queue.enqueueNDRangeKernel(min_distance_kernel, cl::NullRange, cl::NDRange(compute_units), cl::NullRange, NULL, &end_event);
	checkErr(err, "min_distance_kernel: enqueueNDRangeKernel()");

	end_event.wait();
	min_distance_time += (std::clock() - local_start_time);
	local_start_time = std::clock();

	err = command_queue.enqueueReadBuffer(winner_distance_array_buffer, CL_TRUE, 0,
		sizeof(float)*compute_units, winner_distance_array);
	checkErr(err, "winner_distance_array_buffer: enqueueReadBuffer()");
	err = command_queue.enqueueReadBuffer(winner_index_array_buffer, CL_TRUE, 0,
		sizeof(float)*compute_units, winner_index_array);
	checkErr(err, "winner_index_array_buffer: enqueueReadBuffer()");

	float current_min_value = FLT_MAX;
	int current_min_index = 0;
	// cout << "WIA:\t";
	for (int i = 0; i < compute_units; i++){
		// cout << winner_index_array[i] << ": " << winner_distance_array[i] << "\t";
		if (winner_distance_array[i] < current_min_value){
			current_min_index = winner_index_array[i];
			current_min_value = winner_distance_array[i];
		}
	}
	// cout << endl;
	update_weight_kernel.setArg(3, current_min_index);
	checkErr(err, "update_weight_kernel: kernel(3)");

	min_distance_read_time += (std::clock() - local_start_time);


	// winner_distance_array_buffer and winner_index_array_buffer

	// int *array = (int *)malloc(sizeof(int));
	// err = command_queue.enqueueReadBuffer(winner_index_buffer, CL_TRUE, 0,
	// 	sizeof(int), array);
	// checkErr(err, "winner_index_buffer: enqueueReadBuffer()");
	//cout << "Winner: \t" << array[0] << "\t";

	// err = command_queue.enqueueReadBuffer(distance_map_buffer, CL_TRUE, 0,
	// 	sizeof(float)*map_side_size*map_side_size, distance_map);
	// checkErr(err, "distance_map_buffer: enqueueReadBuffer()");
	// //</OPENCL>

	// for (int distance_index = 0; distance_index < map_side_size*map_side_size; distance_index++){
	// 	if (distance_map[distance_index] < winnerDistance){
	// 		winnerDistance = distance_map[distance_index];
	// 		winner = distance_index;
	// 	}
	// 	//cout << distance_map[distance_index] << "\t";
	// }
	// if (current_min_index != winner){
	// 	// cout << current_min_index << " != " << winner << endl;
	// 	cout << "Pass\t";
	// } else {
	// 	cout << "Fail\t";
	// }
	// cout << "W: " << winner << endl;
	// return winner;
}
// float euclidean_distance(float *a, float *b, int a_start_index, int b_start_index, int vector_size){
float quantisationError(int input_index){
	int winner = 0;
	findWinner(input_index);
	//<OPENCL>
	err = command_queue.enqueueReadBuffer(distance_map_buffer, CL_TRUE, 0,
		sizeof(float)*map_side_size*map_side_size, distance_map);
	checkErr(err, "distance_map_buffer: enqueueReadBuffer()");
	//</OPENCL>
	float winnerDistance = FLT_MAX;
	for (int distance_index = 0; distance_index < map_side_size*map_side_size; distance_index++){
		if (distance_map[distance_index] < winnerDistance){
			winnerDistance = distance_map[distance_index];
			winner = distance_index;
		}
		//cout << distance_map[distance_index] << "\t";
	}

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

int main(int argc, char* argv[]){

	// <INPUT INIT>
	string input_filename;
	if (argc < 3){
		cout << "Standard usage: ./parallel_SOM <input file name> <map side size>" << endl;
	}
	if (argc > 1){
		input_filename = argv[1];
	}
	else {
		cout << "Enter input filename:\t";
		std::cin >> input_filename;
	}
	input = readInputFromFile(input_filename);

	if (argc > 2){
		// argv[2] >> map_side_size;
		map_side_size = atoi(argv[2]);
	}
	else {
		cout << "Enter map dimension (x and y dimenions will be equal): \t";
		std::cin >> map_side_size;
	}

	min = 0;
	max = 10000;
	range = max - min;
	previous_map = (float *)malloc(sizeof(float)*map_side_size*map_side_size*input_vector_length);
	gauss_value_list = (float *)malloc(sizeof(float)*map_side_size);
	// distance_map = (float *)malloc(sizeof(float)*map_side_size*map_side_size);

	// input = initialiseRandomArray(input_size, input_vector_length);
	// writeToFile(input, input_size, "input.dat");
	// input = initialiseClusteredArray(input_size, input_vector_length, input_data_clusters);
	// writeToFile(input, input_size, "input_clustered.dat");

	//drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/initial_map.html");
	// </INPUT INIT>

	// <OPENCL>
	vector<cl::Platform> platforms;
	string platform_name;
	cl::Platform::get(&platforms);
	cout << endl;
	for (int i = 0; i < platforms.size(); i++){
		platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platform_name);
		cout << "Platform " << i << ": " << platform_name << endl;
	}
	cout << endl;

	platforms[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platform_name);

	cl_context_properties context_props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
	CPU_context = cl::Context(
		CL_DEVICE_TYPE_CPU,
		context_props,
		NULL,
		NULL,
		&err);
	checkErr(err, "CPU_context()");

	vector<cl::Device> devices;
	devices = CPU_context.getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	devices[0].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units);

	//int chunk_size = map_side_size*map_side_size + (compute_units - (map_side_size*map_side_size)%compute_units) / compute_units;
	int chunk_size;
	if ((map_side_size*map_side_size)% compute_units == 0){
		chunk_size = (map_side_size*map_side_size)/ compute_units;
	}else {
		chunk_size = ((map_side_size*map_side_size) + (compute_units - ((map_side_size*map_side_size)%compute_units)))/compute_units;
	} 

	// distance_map = (float *)malloc(sizeof(float)*map_side_size*map_side_size);
	distance_map = (float *)malloc(sizeof(float)*chunk_size*compute_units);
	for (int i = map_side_size*map_side_size; i < chunk_size*compute_units; i++){
		distance_map[i] = -1;
	}
	winner_distance_array = (float *)malloc(sizeof(float)*compute_units);
	winner_index_array = (int *)malloc(sizeof(int)*compute_units);
	for (int i = 0; i < compute_units; i++){
		winner_distance_array[i] = 0;
		winner_index_array[i] = 0;
	}

	// distance_map_buffer = cl::Buffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
	// 	sizeof(float)*(map_side_size*map_side_size), distance_map, &err);
	distance_map_buffer = cl::Buffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		sizeof(float)*(chunk_size*compute_units), distance_map, &err);
	checkErr(err, "distance_map_buffer");
	input_buffer = cl::Buffer(CPU_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(float)*input_size*input_vector_length, input, &err);
	checkErr(err, "input_buffer");
	gauss_value_list_buffer = cl::Buffer(CPU_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		sizeof(float)*map_side_size, gauss_value_list, &err);
	checkErr(err, "gauss_value_list_buffer");
	
	int *winner_array = (int *)malloc(sizeof(int));
	winner_array[0] = 0;
	winner_index_buffer = cl::Buffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 
		sizeof(int), winner_array, &err);
	checkErr(err, "winner_index_buffer");

	winner_index_array_buffer = cl::Buffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		sizeof(int)*compute_units, winner_index_array, &err);
	checkErr(err, "winner_index_array_buffer");
	winner_distance_array_buffer = cl::Buffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		sizeof(float)*compute_units, winner_distance_array, &err);
	checkErr(err, "winner_distance_array_buffer");


	command_queue = cl::CommandQueue(CPU_context, devices[0],0,&err);
	checkErr(err, "command_queue()");

	// <MANHATTAN_DISTANCE STUFF>
	std::ifstream manhattan_file("manhattan_distance.cl");
	checkErr(manhattan_file.is_open() ? CL_SUCCESS:-1, "manhattan_distance.cl");
	string manhattan_distance_code(std::istreambuf_iterator<char>(manhattan_file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources manhattan_distance_source(1, std::make_pair(manhattan_distance_code.c_str(), manhattan_distance_code.length()+1));
	cl::Program manhattan_distance_prog(CPU_context, manhattan_distance_source);
	err = manhattan_distance_prog.build(devices, "");
	checkErr(err, "Program::build(): manhattan_distance_prog");

	manhattan_distance_kernel = cl::Kernel(manhattan_distance_prog, "manhattan_distance");
	checkErr(err, "manhattan_distance_kernel");	
	manhattan_distance_kernel.setArg(0, input_buffer);
	checkErr(err, "manhattan_distance_kernel: kernel(0)");
	// manhattan_distance_kernel.setArg(1, map_buffer);
	// checkErr(err, "kernel(1)");
	manhattan_distance_kernel.setArg(2, distance_map_buffer);
	checkErr(err, "manhattan_distance_kernel: kernel(2)");
	manhattan_distance_kernel.setArg(3, input_vector_length);
	checkErr(err, "manhattan_distance_kernel: kernel(3)");
	// </MANHATTAN_DISTANCE STUFF>

	// <UPDATE_WEIGHT STUFF>
	std::ifstream update_weight_file("update_weight.cl");
	checkErr(update_weight_file.is_open() ? CL_SUCCESS:-1, "update_weight.cl");
	string update_weight_code(std::istreambuf_iterator<char>(update_weight_file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources update_weight_source(1, std::make_pair(update_weight_code.c_str(), update_weight_code.length()+1));
	cl::Program update_weight_prog(CPU_context, update_weight_source);
	err = update_weight_prog.build(devices, "");
	checkErr(err, "Program::build(): update_weight_prog");

	update_weight_kernel = cl::Kernel(update_weight_prog, "update_weight");
	checkErr(err, "update_weight_kernel");

	// KERNEL ARG 0 (map) ADDED LATER
	update_weight_kernel.setArg(1, input_buffer);
	checkErr(err, "update_weight_kernel: kernel(1)");
	update_weight_kernel.setArg(2, gauss_value_list_buffer);
	checkErr(err, "update_weight_kernel: kernel(2)");
	// update_weight_kernel.setArg(3, winner_index_buffer);
	// checkErr(err, "update_weight_kernel: kernel(3)");
	// KERNEL ARG 4 (input_start_index) ADDED LATER
	update_weight_kernel.setArg(5, input_vector_length);
	checkErr(err, "update_weight_kernel: kernel(5)");
	update_weight_kernel.setArg(6, map_side_size);
	checkErr(err, "update_weight_kernel: kernel(6)");

	update_weight_kernel.setArg(7, output_buffer);
	checkErr(err, "update_weight_kernel: kernel(7)");
	// </UPDATE_WEIGHT STUFF>

	// <MIN_DISTANCE STUFF>
	std::ifstream min_distance_file("min_distance.cl");
	checkErr(min_distance_file.is_open() ? CL_SUCCESS:-1, "min_distance.cl");
	string min_distance_code(std::istreambuf_iterator<char>(min_distance_file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources min_distance_source(1, std::make_pair(min_distance_code.c_str(), min_distance_code.length()+1));
	cl::Program min_distance_prog(CPU_context, min_distance_source);
	err = min_distance_prog.build(devices, "");
	checkErr(err, "Program::build(): min_distance_prog");

	min_distance_kernel = cl::Kernel(min_distance_prog, "min_distance");
	checkErr(err, "min_distance_kernel");

	// min_distance_kernel.setArg(0, distance_map_buffer);
	// checkErr(err, "min_distance_kernel: kernel(0)");
	// min_distance_kernel.setArg(1, map_side_size*map_side_size);
	// checkErr(err, "min_distance_kernel: kernel(1)");
	// min_distance_kernel.setArg(2, winner_index_buffer);
	// checkErr(err, "min_distance_kernel: kernel(2)");

	min_distance_kernel.setArg(0, distance_map_buffer);
	checkErr(err, "min_distance_kernel: kernel(0)");
	min_distance_kernel.setArg(1, winner_index_array_buffer);
	checkErr(err, "min_distance_kernel: kernel(1)");
	min_distance_kernel.setArg(2, winner_distance_array_buffer);
	checkErr(err, "min_distance_kernel: kernel(2)");
	min_distance_kernel.setArg(3, chunk_size);
	checkErr(err, "min_distance_kernel: kernel(3)");
	
	// </MIN_DISTANCE STUFF>

	// </OPENCL>

	cout << "== Stuff to do..\t ==" << endl
			<< "\t- Make vectors into static arrays\t\t<DONE>" << endl
			<< "\t\t+ Arrays must be one dimensional" << endl
			<< "\t\t+ Fix iteration" << endl
			<< "\t- Add manhattan_distance() \t\t\t<DONE>" << endl
			<< "\t- Separate loops\t\t\t\t<DONE>" << endl
			<< "\t- Set up optimal map finding\t\t\t<DONE>" << endl
			<< "\t\t+ Set up quantisation error checker" << endl
			<< "\t\t+ Set up repeated map building routine" << endl
			<< "\t- Tune gaussian curve\t\t\t\t<DONE>" << endl
			<< "\t- Set up openCL version\t\t\t\t<DONE>" << endl
			<< "\t\t+ Put functions into C code" << endl
			<< "\t\t+ Put functions into separate files" << endl
			<< "\t\t+ Add openCL stuff" << endl
			<< "\t- Gaussian clusters\t\t\t\t" << endl
			<< "\t- Explore neighbourhood functions\t\t" << endl
			<< "\t- Explore distance metrics\t\t\t" << endl
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
			<< "\n\t- Platforms available\t\t\t" << platforms.size() << endl
			<< "\t- Running on\t\t\t\t" << platform_name << endl
			<< "\t- Devices available\t\t\t" << devices.size() << endl
			<< "\t- Max Compute Units (per device)\t" << compute_units << endl
 			<< "==\t\t\t==" << endl;

	int current;
	int *winner = (int *)malloc(sizeof(int));
	int iteration;
	int total_map_values = map_side_size*map_side_size*input_vector_length;
	int total_input_values = input_size*input_vector_length;
	// map = initialiseRandomArray(map_side_size*map_side_size, input_vector_length);
	//for (iteration = 0; !convergent() || iteration == 0; iteration++){
	for (int current_trial = 0; current_trial < trials; current_trial++){
		update_weight_time = 0;
		manhattan_distance_time = 0; 
		min_distance_time = 0;
		min_distance_read_time = 0;
		global_start_time = std::clock();
		map = initialiseRandomArray(map_side_size*map_side_size, input_vector_length);

		// <OPENCL>
		map_buffer = cl::Buffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
			sizeof(float)*map_side_size*map_side_size*input_vector_length, map, &err);
		checkErr(err, "map_buffer");
		
		manhattan_distance_kernel.setArg(1, map_buffer);
		checkErr(err, "manhattan_distance_kernel: kernel(1)");

		update_weight_kernel.setArg(0, map_buffer);
		checkErr(err, "update_weight_kernel: kernel(0)");
		// </OPENCL>
		drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/initial_map.ppm");
		for (int i = 0; i < map_side_size*map_side_size*input_vector_length; i++){
			previous_map[i] = map[i];
		}
		calculateGaussList();
		// <OPENCL>
		err = command_queue.enqueueWriteBuffer(gauss_value_list_buffer, CL_TRUE, 0, map_side_size, gauss_value_list);
		checkErr(err, "enqueueWriteBuffer(): gauss_value_list_buffer");
		// </OPENCL>
		writeToFile(gauss_value_list, map_side_size, "learning_rates.dat");
		time_t start_time = time(0);
		cout << "TRIAL " << current_trial << ": started at " << asctime(localtime(&start_time));
		// cout << "TRIAL " << current_trial << endl;
		for (iteration = 0; iteration < cycle_length*map_side_size; iteration++){
			drawProgessBar(iteration, cycle_length*map_side_size);
			
			//current_time = time(0);
			//cout << "Iteration: " << iteration << "\tNon convergent points: " << non_convergent_points << "\t" << asctime(localtime(&current_time));
			//cout << "Iteration: " << iteration << "\t" << asctime(localtime(&current_time));
			for (int input_index = 0; input_index < total_input_values; input_index = input_index+input_vector_length){
				findWinner(input_index);
				// winner = 1;
				update_weights(input_index, input_vector_length);
				// <OPENCL>
				// err = command_queue.enqueueWriteBuffer(map_buffer, CL_TRUE, 0,input_size*input_vector_length, map);
				// checkErr(err, "enqueueWriteBuffer(): map_buffer");
				// </OPENCL>
			}
			if (iteration%cycle_length==0 && iteration != 0){
				// if (gauss_value > 1){
				// 	gauss_value--;
				// 	cout << "Neighbourhood reduced\t";
				// 	calculateGaussList();
				// }
				// else if (gauss_value >= 0.5){
				// 	gauss_value -= 0.1;
				// 	cout << "Neighbourhood reduced\t";
				// 	calculateGaussList();
				// }
				if (gauss_value_list[1] != 0){
					//cout << "Neighbourhood reduced\t" << endl;
					shuntGaussList();
					// <OPENCL>
					err = command_queue.enqueueWriteBuffer(gauss_value_list_buffer, CL_TRUE, 0, sizeof(float)*map_side_size, gauss_value_list);
					checkErr(err, "enqueueWriteBuffer(): gauss_value_list_buffer");
					// </OPENCL>
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
		int seconds = difftime(time(0), start_time);
		cout << endl << "Finished after: " << seconds << " seconds" << endl;
		cout << "KERNEL EXECUTION:" << endl;
			cout << "\tUpdate Weight:\t\t\t" << (update_weight_time/(double)(std::clock() - global_start_time))*100 << "%" << endl;
			cout << "\tMinimum Distance:\t\t" << (min_distance_time/(double)(std::clock() - global_start_time))*100 << "%" << endl;
			cout << "\tMinimum Distance Read Time:\t" << (min_distance_read_time/(double)(std::clock() - global_start_time))*100 << "%" << endl;
			cout << "\tManhattan Distance:\t\t" << (manhattan_distance_time/(double)(std::clock() - global_start_time))*100 << "%" << endl;
			cout << endl;
			err = command_queue.enqueueReadBuffer(map_buffer, CL_TRUE, 0, sizeof(float)*map_side_size*map_side_size*input_vector_length, map);
		checkErr(err, "map_buffer: enqueueReadBuffer()");
		//cout << "Convergent at iteration " << iteration << "!" << endl;
		//cout << "Completeion at iteration " << iteration << "!" << endl;
		// drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/convergent_map.html");
		float total_quantisation_error = 0;
		for (int input_index = 0; input_index < total_input_values; input_index = input_index+input_vector_length){
			total_quantisation_error += quantisationError(input_index);
		}
		cout << "Average Quantisation Error: " << total_quantisation_error/input_size << endl;
		std::ostringstream convert;   // stream used for the conversion
		convert << current_trial;
		drawMap(map, map_side_size*map_side_size, input_vector_length, "map_draw/map_trial_" + convert.str() + ".ppm");
		writeToFile(map, map_side_size*map_side_size, "map_"+convert.str() + ".dat");
		cout << endl;
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
