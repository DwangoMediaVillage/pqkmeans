//

#include <cmath>
#include <numeric>
#include "native_clustering_sample.h"


double l2distance(std::vector<float> x, std::vector<float> y) {
    assert(x.size() == y.size());
    float dist = 0;
    for (int i = 0; i < x.size(); ++i) {
        dist += std::abs(x[i] - y[i]);
    }
    return dist;
}

double calc_score(std::vector<float> x) {
    return std::accumulate(x.begin(), x.end(), 0);
}


void  NativeClusteringSample::fit_one(std::vector<float> pyvector) {
    if (this->min_vec.size() == 0) {
        this->min_vec = pyvector;
    }
    if (this->max_vec.size() == 0) {
        this->max_vec = pyvector;
    }

    // update
    if (calc_score(pyvector) < calc_score(this->min_vec)) {
        this->min_vec = pyvector;
    }
    if (calc_score(pyvector) > calc_score(this->max_vec)) {
        this->max_vec = pyvector;
    }
}

std::vector<float> NativeClusteringSample::transform_one(std::vector<float> pyvector) {
    std::vector<float> result = {0};
    if (l2distance(pyvector, this->min_vec) < l2distance(pyvector, this->max_vec)) {
        result[0] = 0;
    } else {
        result[0] = 1;
    }
    return result;
}