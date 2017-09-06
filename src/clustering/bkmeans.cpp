//

#include <bitset>
#include "bkmeans.h"
#include <sstream>


BKMeans::BKMeans(unsigned int dimention, unsigned int subspace) {

    switch (dimention) {
        case 32:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal = create_bkmeans<32, 2>();
                    break;
                default:
                    std::ostringstream msg;
                    msg << "(dimention, subspace) = (" << dimention << "," << subspace << " ) is not supported";
                    throw msg;
                    break;
            }
            break;
        default:
            std::ostringstream msg;
            msg << "dimention : " << dimention << " is not supported";
            throw msg;
    }
}

void BKMeans::fit_one(const std::vector<float> &pyvector) {

}

int BKMeans::predict_one(const std::vector<float> &pyvector) {
    return 0;
}



