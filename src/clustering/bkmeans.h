//

#ifndef PQKMEANS_BKMEANS_H
#define PQKMEANS_BKMEANS_H

#include <vector>
#include "i_bkmeans_internal.h"
#include "bkmeans_internal.h"

namespace pqkmeans {
template<size_t N, size_t SUB>
std::unique_ptr<IBKmeansInternal> create_bkmeans(unsigned int k) {
    std::string test = "test";
    return std::unique_ptr<IBKmeansInternal>(new BKmeansInternal<N, SUB>(
            k,
            (unsigned int) 3,
            (unsigned int) 3,
            false,
            test.c_str()
    ));
};

class BKMeans {
private:
    std::unique_ptr<IBKmeansInternal> bKmeansInternal_;
public:
    BKMeans(unsigned int k, unsigned int dimention, unsigned int subspace);

    int predict_one(const std::vector<unsigned int> &pyvector);

    void fit(const std::vector<std::vector<unsigned int>> &pydata);
const std::vector<int> GetAssignments();
};
}


#endif //PQKMEANS_BKMEANS_H
