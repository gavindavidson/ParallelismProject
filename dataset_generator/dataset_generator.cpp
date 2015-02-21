#include <iostream>
#include <cstdlib>
#include <string>
#include <ctime>
#include <sstream>
#include <fstream>
#include <cmath>

int max, min, range;
const double pi = 3.14159265359;

void writeToFile(float *data, int size, int input_vector_length, std::string filename){
	std::cout << "Writing to \"" << filename << "\"..." << std::endl;
	std::string contents = "";
	std::ostringstream convert;
	convert << size; 
	contents += convert.str() + " ";

	convert.str("");
	convert << input_vector_length; 
	contents += convert.str();
	
	for (int i = 0; i < size*input_vector_length; i++){
		if (i%input_vector_length == 0){
			contents += "\n";
		}
		convert.str("");
		convert << data[i]; 
		contents += convert.str() + "\t";
	}
	std::ofstream outfile;
	outfile.open(filename.c_str());
	outfile << contents;
	outfile.close();
}


/*
	Function produces a random dataset within certain predifined ranges
*/
float * initialiseRandomArray(int array_size, int vector_length){
	std::cout << std::endl << "Generating.." << std::endl;
	srand(time(NULL));				// Seed rand() with current time
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
	std::cout << std::endl << "Generating.." << std::endl;
	srand(time(NULL));
	float *output = (float *)malloc(array_size*vector_length*sizeof(float));
	float centre;
	float max_variance = max/20;	// Clusters are to be within 5% of the max value of the centre point
	for (int outer = 0; outer < clusters; outer++){
		centre = (rand()/(float)RAND_MAX) * range + min;
		for (int inner = 0; inner < (array_size * vector_length)/clusters; inner++){
			output[inner + (outer*(array_size * vector_length)/clusters)] = (rand()/(float)RAND_MAX) * max_variance + centre;
			//std::cout << inner + (outer*(array_size * vector_length)/clusters) << " ";
		}
	}
	for (int i = 0; i < (array_size * vector_length)%clusters; i++){
		output[(array_size * vector_length) - 1 - i] = (rand()/(float)RAND_MAX) * range + min;
	}
	return output;
}

/*
	Function produces a random dataset that is purposefully clustered into guassian groups for testing
*/
float* initialiseGaussianClusteredArray(float gauss_value, int spread, int number_of_vectors, int vector_length, int clusters){
	std::cout << std::endl << "Generating.." << std::endl;
	float a = 1.0/(gauss_value*sqrt(2*pi));
	float *gauss_value_count_list = (float *)malloc(sizeof(float)*spread);
	float *output = (float *)malloc(number_of_vectors*vector_length*sizeof(float));
	//cout << "New gauss_value_list:\n[";
	for (int x = 0; x < spread; x++){
		gauss_value_count_list[x] = 1000 * a * exp(-(pow(x/5.0, 2)/(2*pow(gauss_value, 2))));
	}

	// Attempt to normalise the amount of values to be produced into the correct size
	float total = 0;
	for (int i = 0; i < spread; i++){
		total += gauss_value_count_list[i];
	}
	float multiplier = ((number_of_vectors*vector_length)/clusters)/total;
	for (int i = 0; i < spread; i++){
		gauss_value_count_list[i] = gauss_value_count_list[i] * multiplier;
	}

	float centre;
	float max_variance = max/40;
	int cluster_size = (number_of_vectors*vector_length)/clusters;
	int current_index = 0;
	int max_index = number_of_vectors*vector_length;
	for (int current_cluster = 0; current_cluster < clusters; current_cluster++){
		// For every desired cluster, make a centre. A top value in this case.
		centre = (rand()/(float)RAND_MAX) * range + min;
		for (int gaus_list_position = 0; gaus_list_position < spread; gaus_list_position++){
			// For every 'amount' value in the gauss_value_count_list
			for (int inner = 0; inner < gauss_value_count_list[gaus_list_position]; inner += 2){
				// Add the 'amount' specified to the output. If this is not the first position
				// in the list, manipulate the centre value before adding it to the output
				if (current_index < max_index - 1){
					output[current_index++] = ((rand()/(float)RAND_MAX) * max_variance) + centre - (gaus_list_position*(centre/spread));
					output[current_index++] = ((rand()/(float)RAND_MAX) * max_variance) + centre + (gaus_list_position*(centre/spread));
				}
				else if (current_index == max_index - 1){
					output[current_index++] = centre;
					return output;	
				}
				else {
					return output;	
				}
			}
		}
	}
	return output;	
}


int main(int argc, char* argv[]){
	std::cout << "==\tRandom data generator for parallel_SOM\t==" << std::endl;
		std::cout << "Commands" << std::endl;
		std::cout << "\t\"uniform\"\tProduces uniformly distributed random data" << std::endl;
		std::cout << "\t\"cluster\"\tProduces clusters of uniformly distributed random data" << std::endl;
		std::cout << "\t\"gaussian\"\tProduces clusters of gaussian distributed random data" << std::endl;
		std::cout << std::endl;
		std::cout << "\t\"info\"\t\tExplains the output of this program" << std::endl;
		std::cout << "\t\"quit\"\t\tEnds program" << std::endl;
		std::cout << "Enter command:\t\t";

	std::string command;
	std::cin >> command;

	while (command.compare("cluster") != 0 && command.compare("uniform") != 0 && command.compare("gaussian") != 0
			&& command.compare("quit") != 0 && command.compare("info") != 0){
		std::cout << "Enter command:\t\t";
		std::cin >> command;
	}
	if (command.compare("quit") == 0){
		exit(0);
	}
	else if (command.compare("info") == 0){
		std::cout << std::endl;
		std::cout << "This program produces a set of vectors of any dimension and saves them in a specified file.";
		std::cout << " The output is designed to be compatible with \"SOM_PAK\" and \"parallel_SOM\" self organising map packages.";
		std::cout << " The first line of the output contains the dimensionality of the vectors and the number of vectors" << std::endl;
		exit(0);
	}

	int num_vectors, vector_length, clusters, spread;
	std::string filename;

	std::cout << "Number of vectors:\t";
	std::cin >> num_vectors;	
	std::cout << "Vector length:\t\t";
	std::cin >> vector_length;
	std::cout << "Output filename:\t";
	std::cin >> filename;
	std::cout << "Max value:\t\t";
	std::cin >> max;
	std::cout << "Min value:\t\t";
	std::cin >> min;

	range = max - min;

	if (command == "cluster"){
		std::cout << "Clusters\t";
		std::cin >> clusters;
		float *data = initialiseClusteredArray(num_vectors, vector_length, clusters);
		writeToFile(data, num_vectors, vector_length, filename);
	}
	else if (command == "gaussian"){
		std::cout << "Clusters:\t\t";
		std::cin >> clusters;
		std::cout << "Spread:\t\t\t";
		std::cin >> spread;
		// initialiseGaussianClusteredArray(float gauss_value, int spread, int number_of_vectors, int vector_length, int clusters){
		float *data = initialiseGaussianClusteredArray(7.0/10, spread, num_vectors, vector_length, clusters);
		writeToFile(data, num_vectors, vector_length, filename);
	}
	else {
		float *data = initialiseRandomArray(num_vectors, vector_length);
		writeToFile(data, num_vectors, vector_length, filename);
	}

	std::cout << "Done!" << std::endl;
}