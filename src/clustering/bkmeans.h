//

#ifndef PQKMEANS_BKMEANS_H
#define PQKMEANS_BKMEANS_H

#include <vector>
#include "bkmeans_impl.h"


class BKMeans {
    BKmeansImpl<32, 2>* bKmeansImpl;
public:
    BKMeans();

    void fit_one(const std::vector<float> &pyvector);

    int predict_one(const std::vector<float> &pyvector);

};


#endif //PQKMEANS_BKMEANS_H
