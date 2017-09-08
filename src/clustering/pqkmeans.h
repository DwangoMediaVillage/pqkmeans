#ifndef PQKMEANS_PQKMEANS_H
#define PQKMEANS_PQKMEANS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <cfloat>
#include <omp.h>

namespace pqkmeans {

//namespace PQKmeansUtil {
//enum class InitCenterType {
//    RandomPick, Outer
//};
//enum class FindNNType {
//    Table,  Linear, Auto
//};
//}



class PQKmeans {
public:
    std::vector<std::vector<unsigned char>> centroids;
    std::vector<int> assignments;

    PQKmeans(std::vector<std::vector<std::vector<float>>> codewords, int K, int itr);

    int predict_one(const std::vector<float> &pyvector);
    void fit(const std::vector<std::vector<unsigned char>> &pydata);

private:
    std::vector<std::vector<std::vector<float>>> codewords_;
    int K_;
    int itr_;
    std::size_t M_; // the number of subspace

    // [m][k1][k2]: m-th subspace, the L2 squared distance between k1-th and k2-th codewords
    std::vector<std::vector<std::vector<float>>> distance_matrices_among_codewords_;

    float SymmetricDistance(const std::vector<unsigned char> &code1,
                            const std::vector<unsigned char> &code2);

    float L2SquaredDistance(const std::vector<float> &vec1,
                            const std::vector<float> &vec2);

    void InitializeCentroidsByRandomPicking(const std::vector<std::vector<unsigned char>> &codes,
                                            int K,
                                            std::vector<std::vector<unsigned char>> *centroids);

    // Linear search by Symmetric Distance computation. Return the best one (id, distance)
    std::pair<std::size_t, float> FindNNLinear(const std::vector<unsigned char> &query,
                                               const std::vector<std::vector<unsigned char>> &codes);

    // Compute a new centroid from assigned codes. codes: All N codes. selected_ids: selected assigned ids.
    // e.g., If selected_ids=[4, 25, 13], then codes[4], codes[25], and codes[13] are averaged by the proposed sparse voting scheme.
    std::vector<unsigned char> ComputeCentroidBySparseVoting(const std::vector<std::vector<unsigned char>> &codes,
                                                             const std::vector<std::size_t> &selected_ids);

};

} // namespace pqkmeans


#endif // PQKMEANS_PQKMEANS_H
