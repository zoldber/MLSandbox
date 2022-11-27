#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <assert.h>

// Data import libraries
#include "../../../tables/src/tbl_csv.hpp"

// Network libraries
#include "../../nnet_toolkit/primitives.hpp"

#define TRAIN_PATH "samples.csv"
#define LABEL_PATH "labels.csv"

int main(void) {

    auto trainSetTable = new DataTable<float>(TRAIN_PATH, ',');
    auto labelSetTable = new DataTable<float>(LABEL_PATH, ',');

    size_t sampleCount = trainSetTable->rowDim();

    assert(sampleCount = labelSetTable->rowDim());

    auto trainSet = trainSetTable->export2DArray();
    auto labelSet = labelSetTable->export2DArray();

    delete trainSetTable;
    delete labelSetTable;

    std::vector<size_t> layerDims = {2, 3, 2};

    auto nnet = new nnet::Network<float>(layerDims);

    float cost;

    // train network
    for (int i = 0; i < 32000; i++) {
    
        cost = nnet->fitNaive(trainSet, labelSet, sampleCount);

        if (i%100==0) std::cout << cost << std::endl;
        
    }

    int r, c;
    
    float xy[2];

    std::ofstream outFile("boundaryMap.csv");

    for (r = 0; r < 15; r ++) {

        xy[1] = (15.0 - (float)r);

        for (c = 0; c < 15; c ++) {

            xy[0] = (float)c;

            outFile << nnet->predict(xy) << ", ";

        }

        outFile << "\n";

    }

    outFile.close();

    return 0;

}