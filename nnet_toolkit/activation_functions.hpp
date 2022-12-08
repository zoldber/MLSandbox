#include <cmath>    // supports exp()
#include <cstring>  // supports memcpy
// Note: this namespace is intentional. 
// The compiler blobs all aliased namespaces together
namespace nnet {
    
    // Replaced element-wise functions f(x) = y for x[0]->x[n] with array modifiers: 
    // f(x, y, n) | x.size()==y.size()==n & fn(x)->y
    // This was done with the intention of supporting aggregating functions like
    // softmax in future updates. None of these should be modifying in-place.

    enum class ActivationTypes { 
        
        // First enumeration (0) should correspond to "no activation function"
        // and pass inputs unchanged: y = f(x) = x and y' = f'(x) = 1.0 for all x.
        // This enables better notation for initializing a network's input layer
        // (whose declared activation function is discarded anyway), and might see
        // use in network debug scripts with direct applicaiton of weights and biases
        none,
        relu,
        lrelu,
        sigmoid,
        fastSigmoid

    };

    // has the effect of passing weighted inputs directly
    template<typename fp>
    void none(fp * x, fp * y, size_t len) {

        std::memcpy(y, x, len * sizeof(fp));

        return;

    }

    // deriv value of 1.0 assumes no scaling in afns
    template<typename fp>
    void d_none(fp * x, fp * y, size_t len) {

        std::memset(y, 1.0, len * sizeof(fp));

        return;

    }

    template<typename fp>
    void relu(fp * x, fp * y, size_t len) {

        for (size_t i = 0; i < len; i++) {

            y[i] = std::max((fp)0.0, x[i]);

        }

        return; 

    }

    template<typename fp>
    void d_relu(fp * x, fp * y, size_t len) {

        for (size_t i = 0; i < len; i++) {

            y[i] = (x[i] > 0.0) ? 1.0 : 0.0;

        }

        return;

    }

    template<typename fp>
    void lrelu(fp * x, fp * y, size_t len) {

        for (size_t i = 0; i < len; i++) {

            y[i] = (x[i] < 0.0) ? 0.01 * x[i] : x[i];

        }

        return;

    }

    template<typename fp>
    void d_lrelu(fp * x, fp * y, size_t len) {

        for (size_t i = 0; i < len; i++) {

            y[i] = (x[i] < 0.0) ? 0.01 : 1.0;

        }

        return;

    }

    template<typename fp>
    void sigmoid(fp * x, fp * y, size_t len) { 
    
        for (size_t i = 0; i < len; i++) {

            y[i] = 1.0 / (1.0 + std::exp(-x[i])); 

        }

        return;
        
    }

    template<typename fp>
    void d_sigmoid(fp * x, fp * y, size_t len) {

        // d_sig(x) = sig(x)(1 - sig(x))

        sigmoid(x, y, len);

        for (size_t i = 0; i < len; i++) {

            y[i] *= (1.0 - y[i]);

        }

        return;
    }
    
    template<typename fp>
    void fastSigmoid(fp * x, fp * y, size_t len) {

        for (size_t i = 0; i < len; i++) {

            y[i] = x[i] / (1.0 + std::abs(x[i]));

        }

        return;

    }

    template<typename fp>
    void d_fastSigmoid(fp * x, fp * y, size_t len) { 
        
        fp tmp;

        for (size_t i = 0; i < len; i++) {

            tmp = 1.0 + std::abs(x[i]);

            y[i] = 1.0 / (tmp * tmp);

        }

        return;
        
    }


    // assign each layer's activation function and corresponding dv afn
    // once within constructor. Iterating through a list of returns based on the
    // enumerations becomes slow for large networks
    template<class fp>
    class Activation {
        private:

        public:

            // for reference: syntax is type_t (*fptr)(type_t x, ..., type_t y)
            void (*applyFunc)(fp * x, fp * y, size_t len);
            void (*applyDeriv)(fp * x, fp * y, size_t len);

            Activation(ActivationTypes type) {

                switch(type) {

                    case ActivationTypes::relu:
                        applyFunc   = &relu;
                        applyDeriv  = &d_relu;
                        break;

                    case ActivationTypes::lrelu:
                        applyFunc   = &lrelu;
                        applyDeriv  = &d_lrelu;
                        break;

                    case ActivationTypes::linear:
                        applyFunc   = &linear;
                        applyDeriv  = &d_linear;
                        break;
                    
                    case ActivationTypes::sigmoid:
                        applyFunc   = &sigmoid;
                        applyDeriv  = &d_sigmoid;
                        break;

                    case ActivationTypes::fastSigmoid:
                        applyFunc   = &fastSigmoid;
                        applyDeriv  = &d_fastSigmoid;
                        break;

                    default:
                        std::cout << "--\tpassed undefined activation function" << std::endl;
                        std::abort();
                        break;

                }

            }

    };

}