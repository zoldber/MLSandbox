# MLSandbox
## A Collection of Neural Network Primitives and Generic Models Written in C++

#### Summary:

Basic neural networks implemented with an emphasis on modularity. This project was undertaken after a few years of experience in developing and training abstracted ML models (e.g. Python libs / APIs), and serves to foster a deeper understanding of network elements and training behavior. 

#### Notes:

#### 1. Interfacing and Portability

Feeding sample features to a network or configuring parameters should be as discrete as possible, and networks won't expect inputs (features or hyper-parameters) that are wrapped in a particular class or struct \[a]. Further, the network should output a vector of activation values as an array of ordinary type, or, at most, a scalar result derived with minimal complexity (e.g. a max or a min within the prediction vector). _The general aim of these models is explicit interfacing with primitive blocks and data structures, as opposed to abstracting them for breivity_.

\[1.a] layer initialization deviates from this goal slightly by using the following:

```cpp

// layer_t: custom struct that contains:
//  - layer size (in nodes), a size_t variable
//  - layer activation function, an enumerated class correpsonding to a first order activation function

layer_t someGenericLayer = { 8, nnet::ActivationTypes::lrelu }; // layer has 8 neurons and uses leaky ReLU

```

This was a simple, and safe alternative to passing a function pointer directly. It restricts the selection of activation functions to a list defined in nnet_toolkit/activation_functions.hpp and ensures that a layer is initialized with a valid activation function and its correpsonding derivative.

#### 2. Dependencies

None, as non-STL libraries (with the _possible_ exception of parallelizing libs) are avoided entirely. It's worth noting that many network demos in this repo will include custom headers for importing data. This minimizes distraction from network-specific code blocks in the demo program, and all reads are cast as ordinary arrays before being fed into a network. I've also made the repo of each custom import header public for ease of debugging / convenience / reference etc.

- - -

### TODO:

- [X] Generic templates and network primitives (layers, gradients, etc.)

- [X] Demo of basic classification by fitting boundary functions

- [ ] Modules optimized for convolutional neural networks (CNN)

- [ ] Demo DNN, CNN implementations of a 28x28 pixel map classifier (benchmark with MNIST and compare accuracy)

- [ ] Modules to support transformers, and a demo of "attention" in the context of natural language processing

- [ ] Parallelization templates (likely using OpenMPI) for high-throughput feature-processing applications
