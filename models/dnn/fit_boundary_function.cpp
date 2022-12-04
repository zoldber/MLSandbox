#include <iostream>

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
    auto features     = new DataTable<float>(TRAIN_PATH, ',');
    auto labels       = new DataTable<float>(LABEL_PATH, ',');

    float ** trainSet = features->export2DArray();
    float ** labelSet = labels->export2DArray();

    // verify matching set length
    size_t lenTrain = features->rowDim();
    assert(lenTrain == labels->rowDim());

    delete features;
    delete labels;

    // Train network with batched inputs and back-propagation
    float learnRate = 0.00250;
    size_t batch = 0;
    size_t numBatches = 3200;
    size_t randBatch, batchSize = 200;

    // build network by layer as a vector of 'nnet::layer_t' elements,
    // where type 'nnet::layer_t' defines { layerNeuronCount, layerActivFunc }
    auto layers = {
        
        (nnet::layer_t){ 2, nnet::ActivationTypes::sigmoid  },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid  },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid  }     

    };

    // create network and cast layers for handling float
    // (as opposed to a larger but more precise fp type)
    auto dnn = new nnet::Network<float>(layers);

    // title for csv log
    std::cout << "batch size: " << batchSize << " learn rate: " << learnRate << std::endl;

    // headers for csv log
    std::cout << "Batch, Cost" << std::endl;

    dnn->resetNetwork(time(0));

    dnn->setLearnRate(learnRate);

    float cost, bestCost = MAXFLOAT;

    while (batch < numBatches) {

        // choose a contiguous set of samples from the population (by index)
        randBatch = rand() % (lenTrain - batchSize);

        dnn->fitBackProp(&trainSet[randBatch], &labelSet[randBatch], batchSize);

        if (batch%10==0) {

            cost = dnn->populationCost(trainSet, labelSet, lenTrain);

            std::cout << batch << ", " << cost << "\n";

            if (cost < bestCost) {

                dnn->saveLayersAsBestFit();

                bestCost = cost;

            }
            
        }

        batch++;

    }

    dnn->recallBestFitLayers();

    float accuracy = dnn->classifierAccuracy(trainSet, labelSet, lenTrain);

    std::cout << "Accuracy: " << accuracy * 100.0 << "% (not cost)" << std::endl;

    // housekeeping
    for (size_t i = 0; i < lenTrain; i++) {

        delete trainSet[i];
        delete labelSet[i];

    }

    delete trainSet;
    delete labelSet;

    delete dnn;

    return 0;

}
