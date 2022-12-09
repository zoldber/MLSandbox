# MLSandbox
## A Collection of Neural Network Primitives and Generic Models Written in C++

## Summary:

Basic neural networks implemented with an emphasis on modularity. This project was undertaken after a few years of experience in developing and training abstracted ML models (e.g. Python libs / APIs), and serves to foster a deeper understanding of network elements and training behavior. 

<br>

## Notes:

#### 1. Interfacing and Portability

Feeding sample features to a network or configuring parameters should be as discrete as possible, and networks won't expect inputs (features or hyper-parameters) that are wrapped in a particular class or struct. Further, the network should output a vector of activation values as an array of ordinary type, or, at most, a scalar result derived with minimal complexity (e.g. a max or a min within the prediction vector). _The general aim of these models is explicit interfacing with primitive blocks and data structures, as opposed to abstracting them for breivity_.

<br>

#### 2. Templates

Model types like **Network** and their members / methods implement a universal generic denoted **fp** (for "floating point") through the use of heirarchical templates. Precision and performance of a model are largely defined by the representation of the floating point values therein, and specific applications will often warrant the use of a long double over a float, etc.

<br>

#### 3. Dependencies

None, as non-STL libraries (with the _possible_ exception of parallelizing libs) are avoided entirely. It's worth noting that many network demos in this repo will include custom headers for importing data. This minimizes distraction from network-specific code blocks in the demo program, and all reads are cast as ordinary arrays before being fed into a network. I've also made the repo of each custom import header public for ease of debugging / convenience / reference etc.

- - -

### TODO:

- [X] Generic templates and common primitives (activation functions, gradients, etc.)

- [X] Basic, fully-connected, feed-forward nnet as proof of concept for modular network components

- [X] Demo of basic classification with the nnet described above by fitting points within boundary functions

- [ ] Network components optimized for convolutional neural networks (CNNs)

- [ ] Demo(s) of feature-extraction, pooling, and classification with the MNIST set and an ordinary CNN

- [ ] Network components that enable simple implementations of transformers
  
- [ ] Demo of "attention" in the context of natural language processing

- [ ] Parallelization templates (likely using OpenMPI) for high-throughput feature-processing applications

- - -
# Network Topologies and Primitive Structures in nnet \{ }

> #### Topologies:
>    1. Classic, Fully-Connected Neural Network ("*Standard Network*")
>    2. Convolutional Neural Network
>    3. 
>
> #### Components:
>    1. Gradient-Paired Layer
>    2. Kernel-Paired Layer

- - -

## Topologies:

## 1. Standard Network

#### Constructors:

```cpp
// (1) Pass a list of layer definitions to build a new network for training. 

// This builds a "full" network with gradients, save buffers, etc.
nnet::Network(const std::vector<layer_t> layerCfg);
```


```cpp
// (2) Pass a file-path to recall trained network layers.

// File header specifies creation of a "sparse" network with only weights
// and biases for a static model, or a full network for further training.
nnet::Network(const std::string filePath);
```

<br>

#### Examples:

```cpp
// (1) See above.

// A Network's first layer has no activation function, and
// the type specified is arbitrary as it's always dicarded
const auto layers = {

    (nnet::layer_t){ 2, nnet::ActivationTypes::none     },
    (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid  },
    (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid  } 

}

auto demoDNN = new nnet::Network<float>(layers);
```
```cpp
// (2) See above.

auto classifier = new nnet::Network<double>(IRIS_CLASSIFIER_PATH);

auto label = classifier->predict(sample);
```

<br>

#### Notes:

<br>

## 2. Convolutional Network

#### Constructors:
<br>

#### Examples:
<br>

#### Constructors:
<br>

#### Notes:

- - -

# Components

## 1. "Standard" Gradient-Paired Layer

#### Constructors:

```cpp
nnet::Layer(const size_t inp, const size_t out, const ActivationTypes afn);
```

<br>

#### Examples:

| Symbol   | Dim.  | Description            | Stored in Sparse Network |
|----------|-------|------------------------|--------------------------|
| **W**    |[m x n]| Weights                | Yes
| **b**    |[m x 1]| Biases                 | Yes
| **x**    |[n x 1]| Input Vector           | No
| **z**    |[m x 1]| Weighted Inp. Vector   | TBD
| **a**    |[m x 1]| Activations (Output)   | Yes
| **e**    |[m x 1]| Label / Expected Output| No
| **W~G~** |[m x n]| Gradient Weights       | No
| **b~G~** |[m x 1]| Gradient Biases        | No
| **D~BP~**|[m x 1]| Back-Prop. Buffer      | No

Given a "full" layer with dimensions *n* and *m* (input and output size, respectively), 
the following methods describe the relationship of the member data above:

<br>

```cpp
fp * evaluate(const fp * input);
```
1.  ![weighted_inputs](resources/weighted_inputs.png) 
2.  ![activation_step](resources/activation_step.png)
- returns pointer to newly updated array **a** for layer chaining in feed(sample) etc.

<br>

```cpp
void applyGradient(const fp learnRate);
```
1.  ![scale_gradient_W](resources/scale_gradient_W.png)
2.  ![scale_gradient_b](resources/scale_gradient_b.png)
- for "learn rate", *r*, given: ![learn_rate_def](resources/learn_rate_def.png)

<br>

```cpp
void updateOutputCostDerivative(const fp * label);
```

<br>

```cpp
void updateHiddenLayerCostDerivative(Layer<fp> * lastLayer);
```

<br>

```cpp
void applyDerivativeVector(void);
```

<br>

## 2. Kernel-Paired Layer

#### Constructors:

```cpp
// (1) This layer and its corresponding kernel are restricted to squares. 
//     Hence, the constructor expects only two arguments 
nnet::AddMePl0x(const size_t layerSize, const size_t kernelSize);
```

<br>

#### Examples: