//

#include <bitset>
#include <cstdint>
#include "bkmeans.h"

BKMeans::BKMeans() {
    std::string test = "test";
    auto data = new std::vector<std::bitset<32>>();
    this->bKmeansImpl = new BKmeansImpl<32, 2>(
            *data,
            (unsigned int)3,
            (unsigned int)3,
            (unsigned int)3,
            false,
            test.c_str()
    );
}

void BKMeans::fit_one(const std::vector<float> &pyvector) {

}

int BKMeans::predict_one(const std::vector<float> &pyvector) {
    return 0;
}



