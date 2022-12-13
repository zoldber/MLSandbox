#include "../../nnet_toolkit/networks.hpp"

int main(void){

	nnet::layer_t input    = { 2, nnet::ActivationTypes::relu    };
	nnet::layer_t hidden_a = { 3, nnet::ActivationTypes::sigmoid };
	nnet::layer_t hidden_b = { 5, nnet::ActivationTypes::sigmoid };
	nnet::layer_t output =   { 3, nnet::ActivationTypes::sigmoid };

	auto layers = { input, hidden_a, hidden_b, output };

	auto dummy = new nnet::Network<float>(layers);

	dummy->printFeatures();

	delete dummy;

	return 0;
}
