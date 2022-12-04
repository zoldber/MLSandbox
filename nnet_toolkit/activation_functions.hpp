#include <cmath>
// Note: this namespace is intentional. The compiler blobs all aliased
// namespaces together
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

    template<typename fp>
    fp relu(fp x) {

        return std::max((fp)0.0, x);

    }

    template<typename fp>
    fp d_relu(fp x) {

    return (x > 0.0) ? 1.0 : 0.0;

    }

    template<typename fp>
    fp lrelu(fp x) {

        return (x < 0) ? 0.01 * x : x;

    }

    template<typename fp>
    fp d_lrelu(fp x) {

        return (x < 0) ? 0.01 : 1.0;

    }

    template<typename fp>
    fp sigmoid(fp x) { 
    
        return 1.0 / (1.0 + std::exp(-x)); 
        
    }

    template<typename fp>
    fp d_sigmoid(fp x) {

        fp s = sigmoid(x);

        return s * (1.0 - s);

    }
    
    template<typename fp>
    fp fastSigmoid(fp x) {

        return x / (1.0 + std::abs(x));

    }

    template<typename fp>
    fp d_fastSigmoid(fp x) { 
        
        fp val = (1.0 + std::abs(x));

        return 1.0 / (val * val);
        
    }


    // assign each layer's activation function and corresponding dv afn
    // once within constructor. Iterating through a list of returns based on the
    // enumerations becomes slow for large networks
    template<class fp>
    class Activation {
        private:

        public:

            fp (*function)(fp);
            fp (*derivative)(fp);

            Activation(ActivationTypes type) {


                switch(type) {

                    case ActivationTypes::sigmoid:
                        function    = &sigmoid;
                        derivative  = &d_sigmoid;
                        break;

                    case ActivationTypes::fastSigmoid:
                        function    = &fastSigmoid;
                        derivative  = &d_fastSigmoid;
                        break;

                    case ActivationTypes::relu:
                        function    = &relu;
                        derivative  = &d_relu;
                        break;

                    case ActivationTypes::lrelu:
                        function    = &lrelu;
                        derivative  = &d_lrelu;
                        break;
                    
                    default:
                        function    = &sigmoid;
                        derivative  = &d_sigmoid;
                        break;

                }

            }

    };

}