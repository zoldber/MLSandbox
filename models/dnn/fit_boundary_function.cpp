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

int main(void) {

    auto trainSetTable = new DataTable<float>(TRAIN_PATH, ',');
    auto labelSetTable = new DataTable<float>(LABEL_PATH, ',');

    assert(labelSetTable->rowDim() == trainSetTable->rowDim());


    size_t sampleCount = trainSetTable->rowDim();

    auto trainSet = trainSetTable->export2DArray();
    auto labelSet = labelSetTable->export2DArray();

    delete trainSetTable;
    delete labelSetTable;

    std::vector<size_t> layerDims = {2, 3, 3};

    auto nnet = new nnet::Network<float>(layerDims);

    // train network
    std::cout.precision(4);
    std::cout << "Pre-fit accuracy:\t"; 
    std::cout << nnet->classifierAccuracy(trainSet, labelSet, sampleCount) * 100.0 << "%" << std::endl;

    float cost, bestCost = MAXFLOAT;

    size_t randBatch, batchSize = 200, batchMax = sampleCount - batchSize;

    for (int i = 0; i < 32000; i++) {

        randBatch = rand() % batchMax;

        nnet->fitBackProp(&trainSet[randBatch], &labelSet[randBatch], batchSize);

        if (i%100==0) {
            
            cost = nnet->populationCost(trainSet, labelSet, sampleCount);

            if (cost < bestCost) {

                nnet->saveLayersAsBest();

                bestCost = cost;

            }

        }

    }

    nnet->recallBestLayers();

    std::cout << "Post-fit accuracy:\t"; 
    std::cout << nnet->classifierAccuracy(trainSet, labelSet, sampleCount) * 100.0 << "%" << std::endl;

    return 0;

}