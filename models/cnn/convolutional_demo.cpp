#include "../../nnet_toolkit/networks.hpp"

int main(void){

    // convolutional / pooling layers
    auto cpLayers = {

        (nnet::conv_layer_t){ 28, 28, 14, 14, nnet::ActivationTypes::relu },
        (nnet::conv_layer_t){ 14, 14,  8,  8, nnet::ActivationTypes::relu },
        (nnet::conv_layer_t){  8,  8,  3,  3, nnet::ActivationTypes::relu }

    };

    // fully connected layers
    auto fcLayers = {

        (nnet::layer_t){ 2, nnet::ActivationTypes::relu    },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid },
        (nnet::layer_t){ 5, nnet::ActivationTypes::sigmoid },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid }

    };

	auto classifier = new nnet::ConvolutionalNetwork<float>(cpLayers, fcLayers);

	classifier->printFeatures();

	delete classifier;

	return 0;
}