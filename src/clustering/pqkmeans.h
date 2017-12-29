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

namespace pqkmeans {

class PQKMeans {
public:
    PQKMeans(std::vector<std::vector<std::vector<float>>> codewords, int K, int itr, bool verbose);

    int predict_one(const std::vector<unsigned char> &pyvector);
    void fit(const std::vector<unsigned char> &pydata);  // pydata is a long array. pydata.size == N * M

    const std::vector<int> GetAssignments();

    void SetClusterCenters(const std::vector<std::vector<unsigned char>> &centers_new);
    std::vector<std::vector<unsigned char>> GetClusterCenters();
    int K() const { return K_; }
    int Iteration() const { return iteration_; }
    bool Verbose() const { return verbose_; }


private:
    std::vector<std::vector<std::vector<float>>> codewords_;  // codewords for PQ encoding
    int K_;
    int iteration_;
    std::size_t M_; // the number of subspace
    bool verbose_;

    std::vector<std::vector<unsigned char>> centers_;  // centers for clustering.
    std::vector<int> assignments_;  // assignement for each intput vector


    // [m][k1][k2]: m-th subspace, the L2 squared distance between k1-th and k2-th codewords
    std::vector<std::vector<std::vector<float>>> distance_matrices_among_codewords_;

    float SymmetricDistance(const std::vector<unsigned char> &code1,
                            const std::vector<unsigned char> &code2);

    float L2SquaredDistance(const std::vector<float> &vec1,
                            const std::vector<float> &vec2);

    void InitializeCentersByRandomPicking(const std::vector<unsigned char> &codes,  // codes.size == N * M
                                          int K,
                                          std::vector<std::vector<unsigned char>> *centers_);

    // Linear search by Symmetric Distance computation. Return the best one (id, distance)
    std::pair<std::size_t, float> FindNearetCenterLinear(const std::vector<unsigned char> &query,
                                                         const std::vector<std::vector<unsigned char>> &codes);

    // Compute a new cluster center from assigned codes. codes: All N codes. selected_ids: selected assigned ids.
    // e.g., If selected_ids=[4, 25, 13], then codes[4], codes[25], and codes[13] are averaged by the proposed sparse voting scheme.
    std::vector<unsigned char> ComputeCenterBySparseVoting(const std::vector<unsigned char> &codes,  // codes.size == N * M
                                                           const std::vector<std::size_t> &selected_ids);

    // Given a long (N * M) codes, pick up n-th code
    std::vector<unsigned char> NthCode(const std::vector<unsigned char> &long_code, std::size_t n);

    // Given a long (N * M) codes, pick up m-th element from n-th code
    unsigned char NthCodeMthElement(const std::vector<unsigned char> &long_code, std::size_t n, int m);

};

} // namespace pqkmeans


#endif // PQKMEANS_PQKMEANS_H
