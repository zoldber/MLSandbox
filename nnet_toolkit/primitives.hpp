#include "activation_functions.hpp"
#include <iostream>
#include <assert.h>
#include <cstring>  // supports std::memcpy()
#include <vector>   // supports layer init, might implement caching in future
#include <time.h>   // supports srand(time(0)), as called in Layer contructor

#define DEFAULT_LEARN_RATE 0.001
#define H 0.001 // NOTE: this hyperparameter is used *only* in the naive fit method

namespace nnet {    

    // { layer size, activation function }
    typedef struct {size_t n; ActivationTypes afn; } layer_t;

    // each gradient simply contains a weight matrix (W), bias vector (b) equal in dim
    // to its corresponding layer during training. The gradient class is isolated from
    // the layer class because the latter need only store its W and b during execution
    template<class fp>
    class Gradient {

        private:

        public:

            size_t inpSize;
            size_t outSize;

            // stores result of partial deriv computation during fitting by back-propagation
            fp * backPropVector;

            fp ** W;
            fp * b;

            // randomize gradient on init.
            Gradient(size_t inp, size_t out) {

                inpSize = inp;
                outSize = out;

                backPropVector = new fp[out]();

                W = new fp * [inp]();

                for (size_t i = 0; i < inp; i++) W[i] = new fp[out]();

                b = new fp[out]();

            }

            void reset(void) {

                size_t inpNode, outNode;

                for (outNode = 0; outNode < outSize; outNode++) {
                    
                    b[outNode] = 0.0;

                }

                for (inpNode = 0; inpNode < inpSize; inpNode++) {

                    for (outNode = 0; outNode < outSize; outNode++) {

                        W[inpNode][outNode] = 0.0;

                    }

                }

            }

    };

    template<class fp>
    class Layer {

        private:

            fp nodeCost(fp output, fp expected) {

                fp error = expected - output;

                return error * error;

            }

            fp d_nodeCost(fp output, fp expected) {

                return 2.0 * (output - expected);

            }

            void randomizeWeights(int seed) {

                std::srand(seed);

                size_t inpNode, outNode;

                fp tmp, scale = (fp)std::sqrt(inpSize);

                for (inpNode = 0; inpNode < inpSize; inpNode++) {

                    for (outNode = 0; outNode < outSize; outNode++) {

                        // normalizes to (-1, 1)
                        tmp = (2.0 * (fp)std::rand() / (fp)RAND_MAX) - 1.0;

                        W[inpNode][outNode] = tmp / scale;

                    }

                }

            }

            // apply partial derivative vector caclulated during back prop to gradient
            void applyDerivativeVector(void) {

                size_t inpNode, outNode;

                // layer's derivative of cost with respect to weight (updated for each node)
                fp d_costW, d_costB;

                for (outNode = 0; outNode < outSize; outNode++) {

                    // update weights
                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        d_costW = gradient->backPropVector[outNode] * inputValues[inpNode];

                        gradient->W[inpNode][outNode] += d_costW;

                    }

                    // update biases
                    d_costB = 1.0 * gradient->backPropVector[outNode];

                    gradient->b[outNode] += d_costB;

                }

            }

        public:

            Activation<fp> * activation;

            Gradient<fp> * gradient;

            size_t inpSize;
            size_t outSize;

            // store and save these for computation of gradients via back-prop
            fp * inputValues;
            fp * inpWeighted;
            fp * activations;

            fp ** bestWeights;
            fp * bestBiases;

            fp ** W;
            fp * b;

            Layer(size_t inp, size_t out, ActivationTypes afn) {

                inpSize = inp;
                outSize = out;

                W = new fp * [inp]();

                bestWeights = new fp * [inp]();

                for (size_t i = 0; i < inp; i++) {
                    
                    W[i] = new fp[out]();

                    bestWeights[i] = new fp[out]();

                }

                b = new fp[out]();

                bestBiases = new fp[out]();

                inputValues = new fp[inp]();
                inpWeighted = new fp[out]();
                activations = new fp[out]();

                randomizeWeights(time(0));

                gradient = new Gradient<fp>(inp, out);

                activation = new Activation<fp>(afn);

            }

            fp * evaluate(fp * input) {

                size_t inpNode, outNode;

                // buffer
                fp n = 0;

                std::memcpy(inputValues, input, inpSize * sizeof(fp));

                for (outNode = 0; outNode < outSize; outNode++) {

                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        n += (inputValues[inpNode] * W[inpNode][outNode]);

                    }

                    inpWeighted[outNode] = n + b[outNode];

                    activations[outNode] = activation->function(inpWeighted[outNode]);

                }

                return activations;

            }

            void applyGradient(fp learnRate) {

                size_t inpNode, outNode;

                for (outNode = 0; outNode < outSize; outNode++) {

                    // apply weights
                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        W[inpNode][outNode] -= (gradient->W[inpNode][outNode] * learnRate);

                    }

                    // apply biases
                    b[outNode] -= (gradient->b[outNode] * learnRate);

                }

            }

            void updateOutputCostDerivative(fp * label) {

                size_t node;

                // partial derivatives of output layer's cost/actv and actv/inpWeighted
                fp d_cost, d_activation;

                for (node = 0; node < outSize; node++) {

                    d_cost = d_nodeCost(activations[node], label[node]);
                    d_activation = activation->derivative(inpWeighted[node]);

                    gradient->backPropVector[node] = d_cost * d_activation;

                }

                applyDerivativeVector();

            }

            void updateHiddenLayerCostDerivative(Layer<fp> * lastLayer) {

                // new node index, old node index
                size_t nni, oni;

                fp newNodeValue, d_inpWeighted;

                for (nni = 0; nni < outSize; nni++) {

                    newNodeValue = 0;

                    for (oni = 0; oni < lastLayer->outSize; oni++) {

                        d_inpWeighted = lastLayer->W[nni][oni];

                        newNodeValue += (d_inpWeighted * lastLayer->gradient->backPropVector[oni]);

                    }

                    newNodeValue *= activation->derivative(inpWeighted[nni]);

                    gradient->backPropVector[nni] = newNodeValue;

                }

                applyDerivativeVector();

            }

            void saveLayerAsBest(void) {

                size_t inpNode, outNode;

                for (outNode = 0; outNode < outSize; outNode++) {

                    // save weights
                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        bestWeights[inpNode][outNode] = W[inpNode][outNode];

                    }

                    // save biases
                    bestBiases[outNode] = b[outNode];

                }

            }

            void recallBestLayer(void) {

                size_t inpNode, outNode;

                for (outNode = 0; outNode < outSize; outNode++) {

                    // save weights
                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        W[inpNode][outNode] = bestWeights[inpNode][outNode];

                    }

                    // save biases
                    b[outNode] = bestBiases[outNode];

                }

            }


            
    };

    template<class fp>
    class Network {

        private:

            std::vector<Layer<fp> *> layers;

            size_t sizeInp;
            size_t sizeOut;

            fp * outputActivation;

            fp learnRate;

        public:

            //Network(const std::vector<size_t> dimensions) {

            Network(const std::vector<layer_t> layerCfg) {

                learnRate = DEFAULT_LEARN_RATE;

                sizeInp = layerCfg.front().n;
                sizeOut = layerCfg.back().n;

                size_t inp, out;

                for (size_t i = 0; i < (layerCfg.size() - 1); i++) {

                    inp = layerCfg.at(i).n;
                    out = layerCfg.at(i+1).n;

                    layers.push_back( new Layer<fp>(inp, out, layerCfg.at(i).afn) );

                }

                outputActivation = layers.back()->activations;

            }

            // use this constructor for recalling a trained model (i.e. set of <fp> type W, b)
            Network(const std::string fileName) { /*TODO*/ return; }

            // consider input validation (?)
            void setLearnRate(fp rate) {

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

            // runs all samples through network and updates weights via
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

                            layers[l]->W[inpNode][outNode] += H;

                            costDiff = populationCost(samples, labels, sampleCount) - preUpdateCost;

                            layers[l]->W[inpNode][outNode] -= H;
                            
                            layers[l]->gradient->W[inpNode][outNode] = costDiff / H;

                        }

                    }

                    // same as above but exclusively for the layer's bias vector
                    for (bias = 0; bias < sizeOut; bias++) {

                        layers[l]->b[bias] += H;

                        costDiff = populationCost(samples, labels, sampleCount) - preUpdateCost;

                        layers[l]->b[bias] -= H;

                        layers[l]->gradient->b[bias] = costDiff / H;

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
                for (size_t l = 0; l < layers.size(); l++) {
                    
                    layers[l]->applyGradient(learnRate);

                    layers[l]->gradient->reset();

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

            void printFeatures(void) {

                /*  Supported activation functions:

                    sigmoid,
                    fastSigmoid,
                    relu,
                    lrelu
                */

                const std::string activationFuncs[4] = {"Sigmoid", "Fast Sigmoid", "ReLU", "Leaky ReLU"};

                for (size_t i = 0; i < this->layers.size(); i++) {
                    std::cout << (i < this->layers.size() - 1 ? "Hidden" : "Output");
                    std::cout << "\t{   W[ " << this->layers.at(i)->inpSize << " x " << this->layers.at(i)->outSize;
                    std::cout << " ], b[ "<< this->layers.at(i)->outSize << " x 1 ]\t} <- " << activationFuncs[i] << std::endl;
                }

                return;

            }


    };

}
