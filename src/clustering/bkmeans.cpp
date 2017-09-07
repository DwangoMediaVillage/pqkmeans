//

#include <bitset>
#include "bkmeans.h"
#include <sstream>

namespace pqkmeans {
    BKMeans::BKMeans(unsigned int dimention, unsigned int subspace) {

        switch (dimention) {
            case 32:
                switch (subspace) {
                    case 2:
                        this->bKmeansInternal = create_bkmeans<32, 2>();
                        break;
                    case 4:
                        this->bKmeansInternal = create_bkmeans<32, 4>();
                        break;
                    case 8:
                        this->bKmeansInternal = create_bkmeans<32, 8>();
                        break;
                    default:
                        std::ostringstream msg;
                        msg << "(dimention, subspace) = (" << dimention << "," << subspace << " ) is not supported";
                        throw msg.str();
                        break;
                }
                break;
            default:
                std::ostringstream msg;
                msg << "dimention : " << dimention << " is not supported";
                throw msg.str();
        }
    }

    void BKMeans::fit_one(const std::vector<float> &pyvector) {
    }

    void BKMeans::fit(const std::vector<std::vector<unsigned int >> &pydata) {
        this->bKmeansInternal->fit(pydata);
    }

    int BKMeans::predict_one(const std::vector<float> &pyvector) {
        return 0;
    }
}
