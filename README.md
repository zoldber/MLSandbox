# MLSandbox
## A Collection of Neural Network Primitves and Generic Models Written in C++

#### Summary:

Basic neural networks written an emphasis on modularity. This project was undertaken after frequent work in developing and training abstracted ML models (e.g. Python libs / APIs), and serves to foster a deeper understanding of network elements and training behavior. To this end, external libraries (excluding parallelizing libs or STL) are avoided.

#### Notes:

#### 1. Unit Testing

#### 2. Interfacing and Portability

Feeding sample features to a network or configuring parameters should be as discrete as possible, and networks won't expect inputs (features or hyper-parameters) that are wrapped in a particular class or struct. Further, the network should output a vector of activation values as an array of ordinary type, or, at most, a scalar result derived with minimal complexity (e.g. a max or a min within the prediction vector). _The general aim of these models is explicit interfacing with primitive blocks and data structures, as opposed to abstracting them for breivity_.

Network demos in this repo will include custom headers for importing data. This minimizes distraction from network-specific code blocks in the demo program, and all reads are cast as ordinary arrays before being fed into a network. I've also made the repo of each custom import header public for ease of debugging / convenience / reference etc.

### TODO:
