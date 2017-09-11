//

#include <bitset>
#include "./bkmeans.h"

namespace pqkmeans {
BKMeans::BKMeans(unsigned int k, unsigned int dimention, unsigned int subspace, unsigned int iteration, bool verbose) {
    switch (dimention) {
        case 32:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<32, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<32, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<32, 8>(k, iteration, verbose);
                    break;
                default:
                    std::ostringstream msg;
                    msg
                    << "(dimention, subspace) = ("
                    << dimention << "," << subspace
                    << " ) is not supported";
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

const std::vector<int> BKMeans::GetAssignments() {
    return this->bKmeansInternal_->GetAssignments();
};


void BKMeans::fit(const std::vector<std::vector<unsigned int >> &pydata) {
    this->bKmeansInternal_->fit(pydata);
}

int BKMeans::predict_one(const std::vector<unsigned int> &pyvector) {
    return this->bKmeansInternal_->FindNearestCentroid(pyvector);
}
}  // namespace pqkmeans
