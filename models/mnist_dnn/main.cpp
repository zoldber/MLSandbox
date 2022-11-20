#include "read_idx.hpp"
#include "lin_alg.hpp"

#define TEST_FILE "train-images.idx3-ubyte"

int main() {

    auto set = new idx::Set<unsigned char>(TEST_FILE);

    auto dims = set->dims();

    uint32_t I, R, C;

    I = std::get<0>(dims);
    R = std::get<1>(dims);
    C = std::get<2>(dims);

    std::cout << I << ", " << R << ", " << C << std::endl; 

    return 0;

}