#include <iostream>
#include <assert.h>
#include <time.h>   // supports srand(time(0)), as called in Layer contructor
#include <tuple>
#include <cmath>

#define LEARN_RATE 1.1
#define H 0.0025

namespace nnet {    

    // note: template typename is for floating-point representation of data
    template<class fp>
    class Layer {

        private:

            fp relu(fp x) {

                return std::max((fp)0.0, x);

            }

            fp d_relu(fp x) {

                return (x > 0.0) ? 1.0 : 0.0;

            }

            fp sigmoid(fp x) { 
            
                return 1.0 / (1.0 + std::exp(-x)); 
                
            }

            fp d_sigmoid(fp x) {

                fp s = sigmoid(x);

                return s * (1.0 - s);

            }

            fp fastSigmoid(fp x) {

                return x / (1.0 + std::abs(x));

            }

            fp d_fastSigmoid(fp x) { 
                
                return 0.0; // TODO: fix me
                
            }

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

        public:

            size_t inpSize;
            size_t outSize;

            // store and save this for computation of gradients
            fp * inpWeighted;

            // note: allocate this as a buffer on init to avoid memory leaks when training
            // (allocate once, fill on each call to eval() and return the same pointer)
            fp * activations;

            fp ** W;
            fp * b;

            Layer(size_t inp, size_t out) {

                inpSize = inp;
                outSize = out;

                W = new fp * [inp]();

                for (size_t i = 0; i < inp; i++) W[i] = new fp[out]();

                b = new fp[out]();

                inpWeighted = new fp[out]();

                activations = new fp[out]();

                randomizeWeights(time(0));

            }

            fp * evaluate(fp * input) {

                size_t inpNode, outNode;

                // buffer
                fp n = 0;

                for (outNode = 0; outNode < outSize; outNode++) {

                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        n += (input[inpNode] * W[inpNode][outNode]);

                    }

                    activations[outNode] = sigmoid(n + b[outNode]);

                }

                return activations;

            }
            
    };

    // each gradient simply contains a weight matrix (W), bias vector (b) equal in dim
    // to its corresponding layer during training. The gradient class is isolated from
    // the layer class because the latter need only store its W and b during execution
    template<class fp>
    class Gradient {

        private:

        public:

            size_t inpSize;
            size_t outSize;

            fp ** W;
            fp * b;

            // randomize gradient on init.
            Gradient(size_t inp, size_t out) {

                inpSize = inp;
                outSize = out;

                W = new fp * [inp]();

                for (size_t i = 0; i < inp; i++) W[i] = new fp[out]();

                b = new fp[out]();

            }

            // pass this method a layer during gradient descent to update
            // weights / biases using their corresponding cost gradient values
            void applyToLayer(Layer<fp> * layer) {

                size_t inpNode, outNode;

                for (outNode = 0; outNode < outSize; outNode++) {

                    layer->b[outNode] -= (b[outNode] * LEARN_RATE);

                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        layer->W[inpNode][outNode] -= (W[inpNode][outNode] * LEARN_RATE);

                    }

                }

            }

    };

    template<class fp>
    class Network {

        private:

            std::vector<Layer<fp> *> layers;
            std::vector<Gradient<fp> *> gradients;

            size_t sizeInp;
            size_t sizeOut;

            fp * outputActivation;

            fp * calcOutputLayerCostPD(Layer<fp> * layer, fp * expected) {

                fp * res = new fp[layer->outSize];

                fp d_cost, d_actv;

                for (int i = 0; i < layer->outSize; i++) {

                    d_cost = layer->d_nodeCost(layer->activations[i], expected[i]);
                    d_actv = layer->d_sigmoid(layer->inpWeighted[i]);


                }

            }

            void updateAllGradients(fp * sample, fp * label) {

                // this fills every layer's weight matrix, biases, activation vals, etc.
                feed(sample);

                auto outputLayer = layers.back();

                // NOTE: he calls this "nodeValues" but it's the product: {da2/dz2} x {dc/da2} = dc/dz2
                fp * outputLayerCostPD = calcOutputLayerCostPD(outputLayer, label);

            }



        public:

            // use this constructor for training (populate gradients)
            Network(const std::vector<size_t> dimensions) {

                sizeInp = dimensions.front();
                sizeOut = dimensions.back();

                size_t inp, out;

                for (size_t i = 0; i < (dimensions.size() - 1); i++) {

                    inp = dimensions[i];
                    out = dimensions[i+1];

                    layers.push_back(new Layer<fp>(inp, out));
                    gradients.push_back(new Gradient<fp>(inp, out));

                }

                outputActivation = layers.back()->activations;

            }

            // use this constructor for recalling a trained model (i.e. set of <fp> type W, b)
            Network(const std::string fileName) { /*TODO*/ return; }

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

            // runs all samples through network and updates weights via naive
            // (i.e. exhaustive) gradient descent and returns average cost of
            // entire sample set after an iteration of training
            fp fitNaive(fp ** samples, fp ** labels, size_t sampleCount) {

                // point span of tangent during gradient descent; as lim(h->0)
                // (f(x + h) - f(x)) / (h) approaches f'(x) for a given x

                fp costDiff;

                fp preUpdateCost = populationCost(samples, labels, sampleCount);

                size_t i, l, L = layers.size();

                size_t inpNode, outNode, bias;

                assert(layers.size() == gradients.size());

                // update all gradients
                for (l = 0; l < L; l++) {

                    // determine cost gradient[i] for layer[i]'s weight matrix
                    for (inpNode = 0; inpNode < sizeInp; inpNode++) {

                        for (outNode = 0; outNode < sizeOut; outNode++) {

                            layers[l]->W[inpNode][outNode] += H;

                            costDiff = populationCost(samples, labels, sampleCount) - preUpdateCost;

                            layers[l]->W[inpNode][outNode] -= H;
                            
                            gradients[l]->W[inpNode][outNode] = costDiff / H;

                        }

                    }

                    // determine cost gradient[i] for layer[i]'s bias vector
                    for (bias = 0; bias < sizeOut; bias++) {

                        layers[l]->b[bias] += H;

                        costDiff = populationCost(samples, labels, sampleCount) - preUpdateCost;

                        layers[l]->b[bias] -= H;

                        gradients[l]->b[bias] = costDiff / H;

                    }

                }

                // apply all newly-updated gradients to their respective layers
                for (l = 0; l < L; l++) {
                    
                    gradients[l]->applyToLayer(layers[l]);

                }

                return populationCost(samples, labels, sampleCount);

            }

            // run all samples through network and update weights via gradient descent,
            // but use back propagation of partial derivatives in each layer 
            fp fitViaBackPropagation(fp * sample, fp * label) {

                

                return 0.0;

            }



    };


}