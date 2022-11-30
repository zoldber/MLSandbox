#include <cmath>

// Note: this namespace isn't a mistake! The compiler blobs all common
// namespaces together (apparently this is even somewhat common in STL)
namespace nnet {
    
    // consider a more elegant way of initializing layer-specific activation functions
    // note that a corresponding dv must be simultaneously set if back-prop is
    // to be used during training
    enum class ActivationTypes { 
        
        sigmoid,
        fastSigmoid,
        relu,
        lrelu
        
    };

    // assign each layer's activation function and corresponding dv afn
    // once within constructor. Iterating through a list of returns based on the
    // enumerations becomes slow for large networks
    template<class fp>
    class Activation {
        private:

            fp relu(fp x) {

                return std::max((fp)0.0, x);

            }

            fp d_relu(fp x) {

                return (x > 0.0) ? 1.0 : 0.0;

            }

            fp lrelu(fp x) {

                return (x < 0) ? 0.01 * x : x;

            }

            fp d_lrelu(fp x) {

                return (x < 0) ? 0.01 : 1.0;

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
                
                fp val = (1.0 + std::abs(x));

                return 1.0 / (val * val);
                
            }

        public:

            // passing a function pointer on init seems like an easy
            // alternative but setting the activation fn and its correpsonding
            // dv once via an enumerated keyword in the constructor is
            // more performant and less confusing. Consider alternatives (but
            // not too hard)
            static fp (Activation::*function)(fp);
            static fp (Activation::*derivative)(fp);

            Activation(ActivationTypes type) {

                switch(type) {

                    case ActivationTypes::sigmoid:
                        function    = &Activation::sigmoid;
                        derivative  = &Activation::d_sigmoid;
                        break;

                    case ActivationTypes::fastSigmoid:
                        function    = &Activation::fastSigmoid;
                        derivative  = &Activation::d_fastSigmoid;
                        break;

                    case ActivationTypes::relu:
                        function    = &Activation::relu;
                        derivative  = &Activation::d_relu;
                        break;

                    case ActivationTypes::lrelu:
                        function    = &Activation::lrelu;
                        derivative  = &Activation::d_lrelu;
                        break;
                    
                    default:
                        function    = &Activation::sigmoid;
                        derivative  = &Activation::d_sigmoid;
                        break;

                }

            }

    };

}