#include "read_idx.hpp"
#include "../../nnet_toolkit/primitives.hpp"

#define TRAIN_PATH "../../../../mnist_data/train-images.idx3-ubyte"
#define LABEL_PATH "../../../../mnist_data/train-labels.idx1-ubyte"

int main() {



    auto trainingImages = new idx::Set<unsigned char>(TRAIN_PATH);
    auto trainingLabels = new idx::Set<unsigned char>(LABEL_PATH);

    auto dims = trainingImages->dims();

    std::vector<size_t> layerDims = {4, 5, 3, 2};

    auto nnet = new nnet::Network<float>(layerDims);

    return 0;

}