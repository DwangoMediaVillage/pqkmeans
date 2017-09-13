//

#include <bitset>
#include "./bkmeans.h"

namespace pqkmeans {
BKMeans::BKMeans(unsigned int k, unsigned int dimention, unsigned int subspace, unsigned int iteration, bool verbose) {
    switch (dimention) {
        case 8:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<8, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<8, 4>(k, iteration, verbose);
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
        case 16:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<16, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<16, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<16, 8>(k, iteration, verbose);
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
                case 16:
                    this->bKmeansInternal_ = create_bkmeans<32, 16>(k, iteration, verbose);
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
        case 64:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<64, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<64, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<64, 8>(k, iteration, verbose);
                    break;
                case 16:
                    this->bKmeansInternal_ = create_bkmeans<64, 16>(k, iteration, verbose);
                    break;
                case 32:
                    this->bKmeansInternal_ = create_bkmeans<64, 32>(k, iteration, verbose);
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
        case 128:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<128, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<128, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<128, 8>(k, iteration, verbose);
                    break;
                case 16:
                    this->bKmeansInternal_ = create_bkmeans<128, 16>(k, iteration, verbose);
                    break;
                case 32:
                    this->bKmeansInternal_ = create_bkmeans<128, 32>(k, iteration, verbose);
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
        case 256:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<256, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<256, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<256, 8>(k, iteration, verbose);
                    break;
                case 16:
                    this->bKmeansInternal_ = create_bkmeans<256, 16>(k, iteration, verbose);
                    break;
                case 32:
                    this->bKmeansInternal_ = create_bkmeans<256, 32>(k, iteration, verbose);
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
        case 512:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<512, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<512, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<512, 8>(k, iteration, verbose);
                    break;
                case 16:
                    this->bKmeansInternal_ = create_bkmeans<512, 16>(k, iteration, verbose);
                    break;
                case 32:
                    this->bKmeansInternal_ = create_bkmeans<512, 32>(k, iteration, verbose);
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
        case 1024:
            switch (subspace) {
                case 2:
                    this->bKmeansInternal_ = create_bkmeans<1024, 2>(k, iteration, verbose);
                    break;
                case 4:
                    this->bKmeansInternal_ = create_bkmeans<1024, 4>(k, iteration, verbose);
                    break;
                case 8:
                    this->bKmeansInternal_ = create_bkmeans<1024, 8>(k, iteration, verbose);
                    break;
                case 16:
                    this->bKmeansInternal_ = create_bkmeans<1024, 16>(k, iteration, verbose);
                    break;
                case 32:
                    this->bKmeansInternal_ = create_bkmeans<1024, 32>(k, iteration, verbose);
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

std::vector<std::vector<unsigned int>> BKMeans::GetClusterCenters() {
    return *(this->bKmeansInternal_->GetClusterCenters());
}


}  // namespace pqkmeans
