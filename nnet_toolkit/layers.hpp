#pragma once

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

            void updateHiddenLayerCostDerivative(Layer<fp> * prevLayer) {

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

                    for (lastInpNode = 0; lastInpNode < prevLayer->outSize; lastInpNode++) {

                        d_inpWeighted = prevLayer->W[thisNodeInp][lastInpNode];

                        gradientAgg += (d_inpWeighted * prevLayer->gradient->backPropVector[lastInpNode]);

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
    class ConvolutionalLayer {

        private:

        public:

            

    };

}
