#include <fstream>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include "types.h"

#define MAX_BUF_LEN 256

class DataSet {

    public:

        std::vector<Sample *> samples;

        DataSet(const std::string filename, const unsigned int feature_count, const unsigned int label_count) {

            std::string line;
            std::ifstream data_file;

            data_file.open(filename);

            if (data_file.is_open() == false) {
                std::cout << "failed to open '" << filename << "'" << std::endl;
                exit(EXIT_FAILURE);
            }

            unsigned int i, j;

            while(getline(data_file, line)) {

                Sample * sample = new Sample(feature_count, label_count);

                j = 0;

                for (i = 0; i < feature_count; i++) {

                    sample->features[i] = atof(&line[j]);

                    while (j < MAX_BUF_LEN && line[j] != '\0' && line[j] != ',') j++;

                    j += 1;

                }

                for (i = 0; i < label_count; i++) {

                    sample->label[i] = atof(&line[j]);

                    while (j < MAX_BUF_LEN && line[j] != '\0' && line[j] != ',') j++;

                    j += 1;                    
                }         

                samples.push_back(sample);  

            }            

        }

        void clear(void) {

            for (auto sample : samples) delete sample;

            return;

        }

};

class Layer {

    private:

        double sigmoid(double val) {

            return 1.0 / (1.0 + std::exp(-val));

        }

        double derivative_sigmoid(double val) {

            double activation_result = sigmoid(val);

            return activation_result * (1 - activation_result);

        }

        double node_cost(double output_activation, double expected_output) {

            double error = expected_output - output_activation;

            // return square of difference
            return error * error;

        }

        double derivative_node_cost(double output_activation, double expected_output) {

            return 2 * (output_activation - expected_output);
        }

        // implement an actual SR normal Gaussian dist later
        double ngd_rand(void) {

            double random = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;

            return random;

        }

    public:

        double ** weights;
        double * biases;

        double ** cost_gradient_weights;
        double * cost_gradient_biases;

        unsigned int inp_dim;
        unsigned int out_dim;

        Layer(unsigned int input_dimension, unsigned int output_dimension) {

            inp_dim = input_dimension;
            out_dim = output_dimension;

            unsigned int inp, out;

            // don't randomize these in the constructor, a network might be re-built from saved vals
            // instead use a member function in Network class (e.g. Network.init()) to randomize
            // or Network.load() to recall saved values
            biases = new  double[out_dim];
            weights = new double * [inp_dim]; 
            
            for (int i = 0; i < inp_dim; i++) { weights[i] = new double[out_dim]; }

            cost_gradient_biases = new double[out_dim];
            cost_gradient_weights = new  double * [inp_dim]; 
            
            for (int i = 0; i < inp_dim; i++) { cost_gradient_weights[i] = new double[out_dim]; }

            // randomize weights ONLY before initiating gradient descent
            double scale = 1.0 / sqrt(inp_dim);

            for (inp = 0; inp < inp_dim; inp++) {

                for (out = 0; out < out_dim; out++) {
                    
                    weights[inp][out] = ngd_rand() * scale;

                    printf("%f\n", weights[inp][out]);

                }

            }

            //TODO: add bias part here

        }

        // returns out_dim sized array from inp_dim sized array
        double * evaluate(double * inputs) {

            unsigned int inp_node;
            unsigned int out_node;

            double * result = (double *)calloc(out_dim, sizeof(double));

            for (out_node = 0; out_node < out_dim; out_node++) {

                for (inp_node = 0; inp_node < inp_dim; inp_node++) {
                    result[out_node] += (inputs[inp_node] * weights[inp_node][out_node]) + biases[out_node];
                }

                result[out_node] = sigmoid(result[out_node]);

            }

            return result;

        }

        void apply_gradient(double rate) {

            unsigned int inp, out;

            for (out = 0; out < out_dim; out++) {

                biases[out] -= cost_gradient_biases[out] * rate;

                for (inp = 0; inp < inp_dim; inp++) {

                    weights[inp][out] -= cost_gradient_weights[inp][out] * rate;

                }

            }

            return;

        }

        void print_nodes(void) {

            unsigned int i, j;

            for (i = 0; i < out_dim; i++) {

                for (j = 0; j < inp_dim; j++) {

                    printf("[%+0.4e]", cost_gradient_weights[j][i]);
                    
                }

                printf("\t+ (%+0.4e)", cost_gradient_biases[i]);

                std::cout << std::endl;

            }

            return;
        }

};

class Network {
    
    private:

        std::vector<Layer *> layers;
        
        double h;

        unsigned int inp_size;
        unsigned int out_size;

    public:

        Network(std::vector<unsigned int> dims) {

            h = 0.0001;

            inp_size = dims.front();
            out_size = dims.back();

            for (unsigned int i = 0; i < (dims.size() - 1); i++) {

                layers.push_back(new Layer(dims.at(i), dims.at(i+1)));

            }

        };

        void set_h(double val) {

            h = val;

            return;
        }

        // returns output of final layer (prediction) for a given set of sample (input) features
        double * predict(Sample * sample) {

            double * layer_values = sample->features;

            for (auto layer : layers) layer_values = layer->evaluate(layer_values);

            return layer_values;

        }

        // returns sum of squared(sample.label[i] - output_value[i]) for a single sample
        double cost(Sample * sample) {

            double * prediction = predict(sample);

            double cost_val = 0, error;

            for (unsigned int out = 0; out < out_size; out++) {

                error = (prediction[out] - sample->label[out]);

                cost_val += (error * error);

            }

            return cost_val;

        }

        double avg_cost(std::vector<Sample *> sample_set) {

            if (sample_set.size() == 0) return 0;

            double average_cost = 0;

            for (auto sample : sample_set) {

                average_cost += cost(sample);

            }

            average_cost /= sample_set.size();

            return average_cost;

        }

        // single iteration of gradient descent
        void train(DataSet * data, double rate) {

            double original_cost = avg_cost(data->samples);

            double cost_difference;

            unsigned int inp, out;

            for (auto layer : layers) {

                // compute cost gradient for prediction made with current weight values
                for (inp = 0; inp < layer->inp_dim; inp++) {

                    for (out = 0; out < layer->out_dim; out++) {
                        
                        layer->weights[inp][out] += h;

                        cost_difference = avg_cost(data->samples) - original_cost;

                        layer->weights[inp][out] -= h;

                        layer->cost_gradient_weights[inp][out] = cost_difference / h;

                    }

                }

                // compute the same for biases
                for (out = 0; out < layer->out_dim; out++) {

                    layer->biases[out] += h;

                    cost_difference = avg_cost(data->samples) - original_cost;

                    layer->biases[out] -= h;

                    layer->cost_gradient_biases[out] = cost_difference / h;

                }

            }

            // apply gradient to all layers
            for (auto layer : layers) {

                layer->apply_gradient(rate);

            }

            return;
        }

        void print(void) {

            unsigned int i = 0;

            for (auto layer : layers) {

                std::cout << "\n- - -LAYER: " << i << std::endl;

                layer->print_nodes();

                i++;
            }

            return;

        }

};