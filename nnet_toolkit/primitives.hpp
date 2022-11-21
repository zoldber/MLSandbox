#include <iostream>
#include <tuple>
#include <cmath>

namespace nnet {    

    // note: template typename is for floating-point representation of data
    template<class fp>
    class Layer {

        private:

            fp sigmoid(fp x) { 
                
                return ((fp)2.0 / ((fp)1.0 + (fp)std::exp(-x))) - (fp)1.0; 
                
            }

            fp fastSigmoid(fp x) {

                return x / ((fp)1.0 + (fp)std::abs(x));

            }

            fp nodeCost(fp output, fp expected) {

                fp error = expected - output;

                return error * error;

            }

            fp dNodeCost(fp output, fp expected) {

                return (fp)2.0 * (output - expected);

            }

            fp ** W;
            fp * b;

            size_t inpSize;
            size_t outSize;

        public:

            Layer(size_t inp, size_t out) {

                inpSize = inp;
                outSize = out;

            }

            

    };

    // each gradient simply contains a weight matrix (W), bias vector (b) equal in dim
    // to its corresponding layer during training. The gradient class is isolated from
    // the layer class because the latter need only store its W and b during execution

    template<class fp>
    class Gradient {

        private:

            size_t inpSize;
            size_t outSize;

        public:

            fp ** W;
            fp * b;

            // randomize gradient on init.
            Gradient(size_t inp, size_t out) {

                inpSize = inp;
                outSize = out;

                W = new fp * [inp];

                for (size_t i = 0; i < inp; i++) W[i] = new fp[out];

                b = new fp[out];

            }

    };

    template<class fp>
    class Network {

        private:

            std::vector<Layer<fp> *> layers;
            std::vector<Gradient<fp> *> gradients;

            size_t sizeInp;
            size_t sizeOut;

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

            }

            // use this constructor for recalling pre-trained network
            Network(const std::string fileName) {

                // TODO

                return;

            }



    };


}