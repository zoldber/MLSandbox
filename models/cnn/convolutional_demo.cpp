#include "../../nnet_toolkit/networks.hpp"

int main(void){

    // convolutional / pooling layers
    auto cpLayers = {

        (nnet::conv_layer_t){ {6, 5, 5}, nnet::ActivationTypes::relu, nnet::PoolingTypes::maxPool },
        (nnet::conv_layer_t){ {6, 5, 5}, nnet::ActivationTypes::relu, nnet::PoolingTypes::maxPool },

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