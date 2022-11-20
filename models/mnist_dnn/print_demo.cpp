/*


    DEMO: Prints the first 5 hand-drawn chars as 28x28 pixelmaps in terminal


*/

#include "read_idx.hpp"

#define TEST_FILE "train-images.idx3-ubyte"
#define LABEL_FILE "train-labels.idx1-ubyte"

int main() {

    auto set = new idx::Set<unsigned char>(TEST_FILE);

    auto lbl = new idx::Set<unsigned char>(LABEL_FILE);

    auto dims = set->dims();

    uint32_t I, R, C;
    uint32_t i, r, c;
    uint32_t x, y, z;

    I = std::get<0>(dims);
    R = std::get<1>(dims);
    C = std::get<2>(dims);

    std::cout << "{ " << I << ", " << R << ", " << C << " }" << std::endl; 
    //std::cout << "{ " << x << ", " << y << ", " << z << " }" << std::endl; 

    unsigned char * symbol;

    for (i = 0; i < 5; i++) {

        symbol = set->item(i);

        for (r = 0; r < R; r++) {

            for (c = 0; c < C; c++) {

                printf((symbol[(r * C) + c] < 128) ? " " : "#");

            }

            puts("");

        }

        printf("symbol: 0x%02X\n", lbl->item(i)[0]);

        puts("\n\n----------------------------\n");

    }


    return 0;

}