#include <cmath>

// see: https://www.researchgate.net/publication/341560698/figure/fig4/AS:893814923337729@1590113492870/Architecture-of-the-convolutional-neural-network-CNN-used-in-the-indoor-Wi-Fi.png
// and: https://en.wikipedia.org/wiki/Convolutional_neural_network#/media/File:Comparison_image_neural_networks.svg
// and: https://archive.vn/vHPpv for back prop. equations, examples

namespace nnet {

    enum class PoolingTypes {
        maxPool,
        minPool,
        avgPool
    };
    
    // Convolution layer and pooling layer are members of 
    template<typename fp>
    void _maxPool(void) {

        return;

    }

    // Assign each layer's activation function and corresponding dv afn
    // once within constructor. Iterating through a list of returns based on the
    // enumerations becomes slow for large networks
    template<class fp>
    class PoolingFunction {
        private:

            // Should be protected by private scope and returned by method call
            PoolingTypes poolingType;

        public:

            // For reference: syntax is type_t (*fptr)(type_t x, ..., type_t y)
            void (*compute)(fp ** x, fp ** y, size_t len);

            Activation(ActivationTypes type) {

                this->functionType = type;

                switch(functionType) {

                    case ActivationTypes::none:
                        applyFunc   = &_none;
                        applyDeriv  = &_d_none;
                        break;

                    case ActivationTypes::tanh:
                        applyFunc   = &_tanh;
                        applyDeriv  = &_d_tanh;
                        break;

                    case ActivationTypes::relu:
                        applyFunc   = &_relu;
                        applyDeriv  = &_d_relu;
                        break;

                    case ActivationTypes::lrelu:
                        applyFunc   = &_lrelu;
                        applyDeriv  = &_d_lrelu;
                        break;
                    
                    case ActivationTypes::sigmoid:
                        applyFunc   = &_sigmoid;
                        applyDeriv  = &_d_sigmoid;
                        break;

                    case ActivationTypes::fastSigmoid:
                        applyFunc   = &_fastSigmoid;
                        applyDeriv  = &_d_fastSigmoid;
                        break;

                    default:
                        
                        // This isn't a practical case but abort anyway
                        abort();

                        break;

                }

            }

            int type(void) { 
                
                int functionType = (int)this->functionType; 

                return functionType;
                
            }

    };

}
