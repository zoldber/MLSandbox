#include <iostream>

#include <assert.h>

#include <string>

// Data import libraries
#include "../../../tables/src/tbl_csv.hpp"

// Network libraries
#include "../../nnet_toolkit/primitives.hpp"

// File paths to training features and labels, respectively
#define TRAIN_PATH "test_data/samples.csv"
#define LABEL_PATH "test_data/labels.csv"

// File path for exporting boundary maps as csv files of colors
#define CLASSIFIER_RESULT_PATH "../../boundary_planes/"

class boundaryPlane {

    private:

        nnet::Network<float> * network;

        size_t x, y, i;
        float coords[2]; // x, y

    public:

        boundaryPlane(nnet::Network<float> * network) {

            this->network = network;

        }

        void generateNew(const std::string fileName) {

            // 0x00RRGGBB
            uint32_t cellColor;

            float * results;

            FILE * fp = fopen(fileName.c_str(), "w");

            for (y = 0; y < 16; y++) {

                coords[1] = (float)y;

                for (x = 0; x < 16; x++) {

                    coords[0] = (float)x;

                    cellColor = 0x0;

                    results = network->feed(coords);

                    fprintf(fp, "#");

                    fprintf(fp, "%02X", (uint32_t)(results[0] * (float)0xFF));
                    fprintf(fp, "%02X", (uint32_t)(results[1] * (float)0xFF));
                    fprintf(fp, "%02X", (uint32_t)(results[2] * (float)0xFF));

                    // numpy struggles to read comma line-terminations 
                    // (e.g. csv line: "a,b,c, " -> [[a, b, c, NaN]])
                    // file << network->predict(coords);
                    fprintf(fp, (x < 15 ? ", " : "\n"));

                }

            }

            fclose(fp);

            return;

        }

};

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

    // Train network with batched inputs and back-propagation.
    // These hyperparameter values chosen after limited experimentation.
    float learnRate = 0.001250;
    size_t batch = 0;
    size_t numBatches = 10000;
    size_t randBatch, batchSize = 400;
    size_t saveBestFreq = 80;

    // Build network by layer as a vector of 'nnet::layer_t' elements,
    // where type 'nnet::layer_t' defines { layerNeuronCount, layerActivFunc }
    const auto layers = {
        
        (nnet::layer_t){ 2, nnet::ActivationTypes::none  },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid  },
        (nnet::layer_t){ 5, nnet::ActivationTypes::sigmoid  },
        (nnet::layer_t){ 3, nnet::ActivationTypes::sigmoid  }     

    };

    // Create network and cast layers for handling float
    // (as opposed to a larger but more precise fp type)
    auto dnn = new nnet::Network<float>(layers);

    dnn->printFeatures();

    auto plane = new boundaryPlane(dnn);

    size_t resCount = 0;

    std::string resName;

    dnn->setLearnRate(learnRate);

    float accuracy, cost, bestCost = MAXFLOAT;

    accuracy = dnn->classifierAccuracy(trainSet, labelSet, lenTrain);

    std::cout << "Init. accuracy:\t" << accuracy * 100.0 << "%" << std::endl;

    while (batch < numBatches) {

        // choose a contiguous set of samples from the population (by index)
        randBatch = rand() % (lenTrain - batchSize);

        dnn->fitBackProp(&trainSet[randBatch], &labelSet[randBatch], batchSize);

        if (batch%saveBestFreq==0) {

            cost = dnn->populationCost(trainSet, labelSet, lenTrain);

            if (cost < bestCost) {

                dnn->saveLayersAsBestFit();

                bestCost = cost;

                resName = CLASSIFIER_RESULT_PATH;
                resName += "bpc_" + std::to_string(resCount) + ".csv";

                plane->generateNew(resName);

                resCount++;

            }

        }

        batch++;

    }

    dnn->recallBestFitLayers();

    accuracy = dnn->classifierAccuracy(trainSet, labelSet, lenTrain);

    std::cout << "Final accuracy: " << accuracy * 100.0 << "%" << std::endl;

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
