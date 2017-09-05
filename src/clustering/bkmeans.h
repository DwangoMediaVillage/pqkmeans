//

#ifndef PQKMEANS_BKMEANS_H
#define PQKMEANS_BKMEANS_H

#include <vector>
#include "i_bkmeans_internal.h"
#include "bkmeans_internal.h"


class BKMeans {
private:
    IBKmeansInternal *bKmeansInternal;
public:
    BKMeans();

    void fit_one(const std::vector<float> &pyvector);

    int predict_one(const std::vector<float> &pyvector);
};


#endif //PQKMEANS_BKMEANS_H
