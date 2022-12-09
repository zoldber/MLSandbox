#include "activation_functions.hpp"
#include <iostream>
#include <assert.h>
#include <cstring>  // supports std::memcpy()
#include <vector>   // supports layer init, might implement caching in future

#define DEFAULT_LEARN_RATE 0.001

namespace nnet {    

    // { layer size, activation function }
    typedef struct { size_t n; ActivationTypes afn; } layer_t;

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

            Gradient(const size_t inp, const size_t out) {

                inpSize = inp;
                outSize = out;

                backPropVector = new fp[outSize]();

                W = new fp * [inpSize]();

                for (size_t i = 0; i < inpSize; i++) {
                    
                    W[i] = new fp[outSize]();

                }

                b = new fp[outSize]();

            }

            ~Gradient(void) {

                for (size_t i = 0; i < inpSize; i++) {
                    
                    delete W[i];

                }

                delete W;

                delete b;

                delete backPropVector;

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

            fp nodeCost(const fp output, const fp expected) {

                fp error = expected - output;

                return error * error;

            }

            fp d_nodeCost(const fp output, const fp expected) {

                return 2.0 * (output - expected);

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

            Layer(const size_t inp, const size_t out, const ActivationTypes afn) {

                inpSize = inp;
                outSize = out;

                W = new fp * [inpSize]();
                bestWeights = new fp * [inpSize]();

                for (size_t i = 0; i < inpSize; i++) {
                    
                    W[i] = new fp[outSize]();

                    bestWeights[i] = new fp[outSize]();

                }

                b = new fp[outSize]();
                bestBiases = new fp[outSize]();

                inputValues = new fp[inpSize]();
                inpWeighted = new fp[inpSize]();
                activations = new fp[inpSize]();

                gradient = new Gradient<fp>(inpSize, outSize);

                activation = new Activation<fp>(afn);

            }

            ~Layer(void) {

                for (size_t i = 0; i < inpSize; i++) {
                    
                    delete W[i];

                    delete bestWeights[i];

                }

                delete b;

                delete bestBiases;

                delete inputValues;
                delete inpWeighted;
                delete activations;


                delete gradient;

            }

            // input[], inpWeighted[], and activations[] are allocated outside
            // of this function within the constructor, and the pointer return
            // of fp * evaluate(vals) is 1/2 notational and 1/2 debug-friendly
            fp * evaluate(const fp * input) {

                size_t inpNode, outNode;

                // buffer
                fp n = 0;

                std::memcpy(inputValues, input, inpSize * sizeof(fp));

                for (outNode = 0; outNode < outSize; outNode++) {

                    for (inpNode = 0; inpNode < inpSize; inpNode++) {

                        n += (inputValues[inpNode] * W[inpNode][outNode]);

                    }

                    inpWeighted[outNode] = n + b[outNode];

                    // WORKING: activations[outNode] = activation->function(inpWeighted[outNode]);

                }

                activation->applyFunc(inpWeighted, activations, outSize);

                return activations;

            }

            void applyGradient(const fp learnRate) {

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

            void updateOutputCostDerivative(const fp * label) {

                // computes first "backPropVector" from the output layer as the product of
                // partial derivatives of output layer's cost/actv and actv/inpWeighted

                // given:
                //      label[] : expected output vals
                //      out[]   : output values, out[] afn()
                //      afn([]) : activation function
                //      afn'([]): derivative of afn([])
                //      cost(,) : cost function of an output val wrt expected val
                //      cost'(,): derivative of cost() for the inputs described above
                //      W[][]   : layer weights
                //      W_i[]   : weighted input vector = matmul(W[][], i[])
                //      a[]     : activation vector = afn(W_i[])
                //      G.v[]   : gradient back-prop. vector           
                //      d_c[]   : derivNodeCost(output[], expected[])
                //      - - - - - - - - - - - - - - - -
                //      Result (original):
                //          for i { G.v[i] = cost'(out[i], label[i]) * afn'(W_i[i]) }
                //
                //      Update (after implementing afns as lambdas):
                //          activation->applyDeriv(W_i, G.v, numNodes)
                //          for i { G.v[i] *= cost'(out[i], label[i]) }

                activation->applyDeriv(inpWeighted, gradient->backPropVector, outSize);

                for (size_t i = 0; i < outSize; i++) {

                    gradient->backPropVector[i] *= d_nodeCost(activations[i], label[i]);

                }

                applyDerivativeVector();

            }

            void updateHiddenLayerCostDerivative(Layer<fp> * lastLayer) {

                // a generalization of the operation above, performs the following steps:

                // given:
                //      thisLyr : layer in which this method is called
                //      lastLyr : ptr to "previous" (normally subsequent) layer in back-prop
                //      W[][]   : weights (same type for both layers but dims might vary) 
                //      G.v[]   : back-prop vector (see previous function desc.)
                //      tmpAgg  : gradient aggregate variable (scalar)
                //      - - - - - - - - - - - - - - - -
                //      Update:
                //          activation->applyDeriv(W_i, G.v, numNodes)
                //
                //          get x,y as lastLyr->W[x][y]
                //
                //          for x {
                //
                //              reset tmpAgg to 0
                //
                //              for y { tmpAgg += ( lastLyr->W[x][y] * lastLyr.G.v[y] }
                //              
                //              thisLyr->G.v[x] *= tmpAgg
                // 
                //          }

                // indices for this layer's node and last, respectively,
                // for use in computing the back prop vector's aggregate
                size_t thisNodeInp, lastInpNode;

                fp gradientAgg;     // tmp for summation of the partial derivs computed over last layer
                fp d_inpWeighted;   // tmp for deriv of W_i computed in last layer

                activation->applyDeriv(inpWeighted, gradient->backPropVector, outSize);

                for (thisNodeInp = 0; thisNodeInp < outSize; thisNodeInp++) {

                    gradientAgg = 0;

                    for (lastInpNode = 0; lastInpNode < lastLayer->outSize; lastInpNode++) {

                        d_inpWeighted = lastLayer->W[thisNodeInp][lastInpNode];

                        gradientAgg += (d_inpWeighted * lastLayer->gradient->backPropVector[lastInpNode]);

                    }

                    gradient->backPropVector[thisNodeInp] *= gradientAgg;

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

            void randomizeWeights(const int seed) {

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
                    relu,
                    lrelu,
                    sigmoid,
                    fastSigmoid
                */

                const std::string activationFuncs[5] = {"None", "ReLU", "Leaky ReLU", "Sigmoid", "Fast Sigmoid"};

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

}
