#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <assert.h>

// Data import libraries
#include "../../../tables/src/tbl_csv.hpp"

// Network libraries
#include "../../nnet_toolkit/primitives.hpp"

#define TRAIN_PATH "test_data/samples.csv"
#define LABEL_PATH "test_data/labels.csv"

using namespace std;

int main(void) {

    auto features   = new DataTable<float>(TRAIN_PATH, ',');
    auto labels     = new DataTable<float>(LABEL_PATH, ',');

    float ** trainSet = features->export2DArray();
    float ** labelSet = labels->export2DArray();

    size_t lenTrain = features->rowDim();

    assert(lenTrain == labels->rowDim());

    nnet::layer_t inputLayer    = { 2, nnet::ActivationTypes::sigmoid };
    nnet::layer_t hiddenLayer   = { 3, nnet::ActivationTypes::sigmoid };
    nnet::layer_t outputLayer   = { 3, nnet::ActivationTypes::sigmoid };

    auto layers = {inputLayer, hiddenLayer, outputLayer};

    auto nnet = new nnet::Network<float>(layers);

    // train network
    std::cout.precision(4);
    std::cout << "Pre-fit accuracy:\t"; 
    std::cout << nnet->classifierAccuracy(trainSet, labelSet, lenTrain) * 100.0 << "%" << std::endl;

    float cost, bestCost = MAXFLOAT;

    size_t randBatch, batchSize = 200, batchMax = lenTrain - batchSize;

    for (int i = 0; i < 32000; i++) {

        randBatch = rand() % batchMax;

        nnet->fitBackProp(&trainSet[randBatch], &labelSet[randBatch], batchSize);

        if (i%100==0) {
            
            cost = nnet->populationCost(trainSet, labelSet, lenTrain);

            if (cost < bestCost) {

                nnet->saveLayersAsBest();

                bestCost = cost;

            }

        }

    }

    nnet->recallBestLayers();

    std::cout << "Post-fit accuracy:\t"; 
    std::cout << nnet->classifierAccuracy(trainSet, labelSet, lenTrain) * 100.0 << "%" << std::endl;

    return 0;


}
