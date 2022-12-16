#include "layers.hpp"

namespace nnet {

template<class fp>
    class Network {

        private:

            std::vector<Layer<fp> *> layers;

            size_t sizeInp;
            size_t sizeOut;

            fp * outputActivation;

            fp learnRate;

        public:

            Network(const std::vector<layer_t> layerCfg) {

                learnRate = DEFAULT_LEARN_RATE;

                sizeInp = layerCfg.front().n;
                sizeOut = layerCfg.back().n;

                size_t inp, out;

                for (size_t i = 0; i < (layerCfg.size() - 1); i++) {

                    inp = layerCfg.at(i).n;
                    out = layerCfg.at(i+1).n;

                    layers.push_back( new Layer<fp>(inp, out, layerCfg.at(i+1).afn) );

                }

                outputActivation = layers.back()->activations;

                // finally, initilize layers and correpsonding gradients
                resetNetwork(time(0));

                return;

            }

            // use this constructor for recalling a trained model (i.e. set of <fp> type W, b)
            Network(const std::string fileName) { /*TODO*/ return; }

            ~Network(void) {

                for (auto layer : layers) delete layer;

                return;

            }

            // consider input validation (?)
            void setLearnRate(const fp rate) {

                learnRate = rate;

            } 

            // returns loss of given sample provided a given label
            fp sampleCost(fp * sample, fp * label) {

                fp * prediction = feed(sample);

                fp cost = 0.0;

                for (size_t i = 0; i < sizeOut; i++) {

                    // cost is summation of squares of error
                    cost += std::pow((prediction[i] - label[i]), 2);

                }

                return cost;

            }

            // returns average loss of all samples in data set
            fp populationCost(fp ** samples, fp ** labels, size_t sampleCount) {

                fp cost = 0.0;

                for (size_t i = 0; i < sampleCount; i++) {

                    cost += sampleCost(samples[i], labels[i]);

                }

                cost /= (fp)sampleCount;

                return cost;

            }

            // returns pointer to network's output nodes (raw activation values)
            fp * feed(fp * sample) {

                fp * nodeVals = sample;

                for (Layer<fp> * layer : layers) {

                    nodeVals = layer->evaluate(nodeVals);

                }

                return nodeVals;

            }

            // returns classification prediction as the max output node activation
            int predict(fp * sample) {

                fp * result = feed(sample);

                int maxNode = 0;

                for (size_t outNode = 1; outNode < sizeOut; outNode++) {

                    if (result[outNode] > result[maxNode]) {

                        maxNode = outNode;

                    }

                }

                return maxNode;

            }

            // Runs all samples through network and updates weights via
            // exhaustive gradient descent and returns average cost of
            // entire sample set after an iteration of training
            fp fitSimple(fp ** samples, fp ** labels, size_t sampleCount) {

                // point span of tangent during gradient descent; as lim(h->0)
                // (f(x + h) - f(x)) / (h) approaches f'(x) for a given x

                fp costDiff;

                fp preUpdateCost = populationCost(samples, labels, sampleCount);

                size_t i, l, L = layers.size();

                size_t inpNode, outNode, bias;

                // update all gradients
                for (l = 0; l < L; l++) {

                    // determine cost gradient[i] for layer[i]'s weight matrix using a cost function
                    // nudge of span H to approximate the local slope and update gradient
                    for (inpNode = 0; inpNode < sizeInp; inpNode++) {

                        for (outNode = 0; outNode < sizeOut; outNode++) {

                            layers[l]->W[inpNode][outNode] += this->learnRate;

                            costDiff = populationCost(samples, labels, sampleCount) - preUpdateCost;

                            layers[l]->W[inpNode][outNode] -= this->learnRate;
                            
                            layers[l]->gradient->W[inpNode][outNode] = costDiff / this->learnRate;

                        }

                    }

                    // same as above but exclusively for the layer's bias vector
                    for (bias = 0; bias < sizeOut; bias++) {

                        layers[l]->b[bias] += learnRate;

                        costDiff = populationCost(samples, labels, sampleCount) - preUpdateCost;

                        layers[l]->b[bias] -= learnRate;

                        layers[l]->gradient->b[bias] = costDiff / learnRate;

                    }

                }

                // apply all newly-updated gradients to their respective layers
                for (l = 0; l < L; l++) {
                    
                    layers[l]->applyGradient(learnRate);

                }

                return populationCost(samples, labels, sampleCount);

            }

            // run all samples through network and update weights via gradient descent,
            // but use back propagation of partial derivatives in each layer 
            void fitBackProp(fp ** samples, fp ** labels, size_t sampleCount) {

                Layer<fp> * outLayer = layers.back();

                int h;  // keep this an int, size_t is unsigned and never < 0 (see while below)

                for (size_t i = 0; i < sampleCount; i++) {

                    // fill layers
                    feed(samples[i]);

                    outLayer->updateOutputCostDerivative(labels[i]);

                    h = layers.size() - 1; // recall that the first "layer" in the nnet isn't actually a class: layer

                    // for all hidden layers
                    while (h --> 0) {

                        //std::cout << "curr: inpSize=" << layers.at(h)->inpSize << ", outSize=" << layers.at(h)->outSize << std::endl;

                        layers.at(h)->updateHiddenLayerCostDerivative(layers.at(h + 1));

                    }

                }

                // apply all newly-updated gradients to their respective layers
                for (auto layer : layers) {

                    layer->applyGradient(learnRate);
                    layer->gradient->reset();

                }

                return;

            }

            fp classifierAccuracy(fp ** samples, fp ** labels, size_t sampleCount) {

                fp correct = 0.0, possible = (fp)sampleCount;

                int prediction;

                for (size_t i = 0; i < sampleCount; i++) {

                    prediction = predict(samples[i]);

                    if (labels[i][prediction] == 1.0) correct += 1.0;

                }

                return correct / possible;

            }

            void saveLayersAsBestFit(void) {

                for (auto layer : layers) layer->saveLayerAsBest();

                return;

            }

            void recallBestFitLayers(void) {

                for (auto layer : layers) layer->recallBestLayer();

                return;

            }

            void resetNetwork(int seed) {

                for (auto layer : layers) {

                    layer->randomizeWeights(seed);
                    layer->gradient->reset();
                    // fixes bug wherein recallBestFitLayers() can be called on a
                    // network without saved layers (each 'best' layer is zero'd)
                    // to erase all randomized weights on start
                    layer->saveLayerAsBest();
                }

            }

            void printFeatures(void) {

                /*  Supported activation functions:
                    none,
                    tanh,
                    relu,
                    lrelu,
                    sigmoid,
                    fastSigmoid
                */

                const std::string activationFuncs[6] = {"None", "tanh", "ReLU", "Leaky ReLU", "Sigmoid", "Fast Sigmoid"};

               std::cout << "Input\t{   W[  N/A  ], b[  N/A  ]\t} <- " << activationFuncs[0] << "\n"; // first layer (input) has no activation function

                for (size_t i = 0; i < this->layers.size(); i++) {
                    std::cout << (i < this->layers.size() - 1 ? "Hidden" : "Output");
                    std::cout << "\t{   W[ " << this->layers.at(i)->inpSize << " x " << this->layers.at(i)->outSize;
                    std::cout << " ], b[ "<< this->layers.at(i)->outSize << " x 1 ]\t} <- ";
                    std::cout << activationFuncs[this->layers.at(i)->activation->type()] << std::endl;
                }

                return;

            }

    };

    // note: this should construct the convolutional stages and append the 
    // fully connected classifier at the end via the ordinary constructor
    template<class fp>
    class ConvolutionalNetwork : public Network<fp> {

        private:

        public:

            ConvolutionalNetwork(const std::vector<conv_layer_t> convPoolCfg, const std::vector<layer_t> fullyConnectedCfg){

                return;

            }

    };

}