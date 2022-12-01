#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <assert.h>

// Data import libraries
#include "../../../tables/src/tbl_csv.hpp"

// Network libraries
#include "../../nnet_toolkit/primitives.hpp"

// File paths to training features and labels, respectively
#define TRAIN_PATH "test_data/samples.csv"
#define LABEL_PATH "test_data/labels.csv"

int main(void) {

    // import data and verify equal set length with assert()
    auto features   = new DataTable<float>(TRAIN_PATH, ',');
    auto labels     = new DataTable<float>(LABEL_PATH, ',');

    float ** trainSet = features->export2DArray();
    float ** labelSet = labels->export2DArray();

    size_t lenTrain = features->rowDim();
    assert(lenTrain == labels->rowDim());

    delete features;
    delete labels;

    // build network by layer as a vector of 'nnet::layer_t' elements,
    // where type 'nnet::layer_t' defines { layerNeuronCount, layerActivFunc }
    auto layers = {
        
        (nnet::layer_t){ 2, nnet::ActivationTypes::sigmoid },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid }     

    };

    // create network and cast layers for handling float
    // (as opposed to a larger but more precise fp type)
    auto dnn = new nnet::Network<float>(layers);

    std::cout << "Pre-fit accuracy:\t"; 
    std::cout << dnn->classifierAccuracy(trainSet, labelSet, lenTrain) * 100.0 << "%" << std::endl;

    // Train network with batched inputs and back-propagation

    float cost, bestCost = MAXFLOAT;

    size_t randBatch, batchSize = 200, batchMax = lenTrain - batchSize;

    for (int i = 0; i < 32000; i++) {

        randBatch = rand() % batchMax;

        dnn->fitBackProp(&trainSet[randBatch], &labelSet[randBatch], batchSize);

        if (i%100==0) {
            
            cost = dnn->populationCost(trainSet, labelSet, lenTrain);

            if (cost < bestCost) {

                dnn->saveLayersAsBest();

                bestCost = cost;

            }

        }

    }

    dnn->recallBestLayers();

    std::cout << "Post-fit accuracy:\t"; 
    std::cout << dnn->classifierAccuracy(trainSet, labelSet, lenTrain) * 100.0 << "%" << std::endl;

    return 0;

}
