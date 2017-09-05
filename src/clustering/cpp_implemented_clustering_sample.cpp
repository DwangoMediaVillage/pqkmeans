//

#include <cmath>
#include <numeric>
#include "cpp_implemented_clustering_sample.h"


double l2distance(const std::vector<float> x, const std::vector<float> y) {
    assert(x.size() == y.size());
    float dist = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        dist += std::abs(x[i] - y[i]);
    }
    return dist;
}

double calc_score(const std::vector<float> x) {
    return std::accumulate(x.begin(), x.end(), 0);
}


void  CppImplementedClusteringSample::fit_one(const std::vector<float> pyvector) {
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

std::vector<float> CppImplementedClusteringSample::transform_one(const std::vector<float> pyvector) {
    std::vector<float> result = {0};
    if (l2distance(pyvector, this->min_vec) < l2distance(pyvector, this->max_vec)) {
        result[0] = 0;
    } else {
        result[0] = 1;
    }
    return result;
}