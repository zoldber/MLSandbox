#include <cmath>
// Note: this namespace is intentional. The compiler blobs all aliased
// namespaces together
namespace nnet {
    
    // Effectively lambda functions f(x, y, l) | x.size()==y.size()==l & fn(x)->y
    // to support aggregate (softmax) as well as element-wise (sigmoid) activations.

    // Note: Should NEVER be in-place (e.g. f(x, l)) with current network training fmt
    enum class ActivationTypes { 
        
        relu,
        lrelu,
        linear,
        sigmoid,
        fastSigmoid
        
    };

    // has the effect of passing weighted inputs directly
    template<typename fp>
    void linear(fp * x, fp * y, size_t len) {

        std::memcpy(y, x, len * sizeof(fp));

        return;

    }

    // deriv value of 1.0 assumes no scaling in afns
    template<typename fp>
    void d_linear(fp * x, fp * y, size_t len) {

        std::memset(y, 1.0, len * sizeof(fp));

        return;

    }

    template<typename fp>
    void relu(fp * x, fp * y, size_t len) {

        for (auto i = 0; i < len; i++) {

            y[i] = std::max((fp)0.0, x[i]);

        }

        return; 

    }

    template<typename fp>
    void d_relu(fp * x, fp * y, size_t len) {

        for (auto i = 0; i < len; i++) {

            y[i] = (x[i] > 0.0) ? 1.0 : 0.0;

        }

        return;

    }

    template<typename fp>
    void lrelu(fp * x, fp * y, size_t len) {

        for (auto i = 0; i < len; i++) {

            y[i] = (x[i] < 0.0) ? 0.01 * x[i] : x[i];

        }

        return;

    }

    template<typename fp>
    void d_lrelu(fp * x, fp * y, size_t len) {

        for (auto i = 0; i < len; i++) {

            y[i] = (x[i] < 0.0) ? 0.01 : 1.0;

        }

        return;

    }

    template<typename fp>
    void sigmoid(fp * x, fp * y, size_t len) { 
    
        for (auto i = 0; i < len; i++) {

            y[i] = 1.0 / (1.0 + std::exp(-x[i])); 

        }

        return;
        
    }

    template<typename fp>
    void d_sigmoid(fp * x, fp * y, size_t len) {

        // d_sig(x) = sig(x)(1 - sig(x))

        sigmoid(x, y, len);

        for (auto i = 0; i < len; i++) {

            y[i] *= (1.0 - y[i]);

        }

        return;
    }
    
    template<typename fp>
    void fastSigmoid(fp * x, fp * y, size_t len) {

        for (auto i = 0; i < len; i++) {

            y[i] = x[i] / (1.0 + std::abs(x[i]));

        }

        return;

    }

    template<typename fp>
    void d_fastSigmoid(fp * x, fp * y, size_t len) { 
        
        fp tmp;

        for (auto i = 0; i < len; i++) {

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