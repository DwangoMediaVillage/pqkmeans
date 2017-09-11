//

#ifndef PQKMEANS_BKMEANS_INTERNAL_H
#define PQKMEANS_BKMEANS_INTERNAL_H


#include <iostream>
#include <sstream>
#include <random>
#include <bitset>
#include <chrono>
#include <climits>
#include <cassert>
#include "i_bkmeans_internal.h"

namespace pqkmeans {
namespace BKmeansUtil {
enum class InitCenterType {
    RandomPick, Random, Outer
};
enum class FindNNType {
    Table, Linear, Auto
};
}

template<size_t N, size_t SUB>
class BKmeansInternal : public IBKmeansInternal {
public:
    BKmeansUtil::FindNNType find_nn_type_;

    BKmeansInternal(unsigned int k,
                    unsigned int iteration,
                    bool verbose = false,
                    BKmeansUtil::InitCenterType init_center_type = BKmeansUtil::InitCenterType::RandomPick
    ) :
            find_nn_type_(BKmeansUtil::FindNNType::Auto), k_(k), iteration_(iteration),
            verbose_(verbose), init_center_type_(init_center_type) {

        // initialize hash tables
        for (unsigned int i = 0; i < N; i += SUB) {
            std::vector<std::vector<int>> table(1UL << SUB);
            this->tables_.push_back(table);
        }
        this->num_subspace_ = this->tables_.size();

        for (unsigned int i = 0; i < N; i++) {
            std::bitset<N> bc;
            bc[i] = 1;
            this->bit_count_map_.push_back(bc);
        }
        this->bit_combinations_ = BitCombinations((unsigned int) SUB);
    }

    void fit(const std::vector<std::vector<unsigned int >> &data) {
        fit(data, std::vector<unsigned int>());
    }

    std::bitset<N> vector2bitset(std::vector<unsigned int> datum) {
        assert(datum.size() == N);
        std::bitset<N> bitset;
        for (std::size_t j = 0; j < N; ++j) {
            bitset[j] = (datum[j] > 0);
        }
        return bitset;

    }

    const std::vector<int> GetAssignments() {
        return assignments;
    };

    void fit(const std::vector<std::vector<unsigned int >> &data,
             std::vector<unsigned int> initialCentroidIndexs = std::vector<unsigned int>()
    ) {
        std::vector<std::bitset<N>> bitset_data;
        for (std::size_t i = 0; i < data.size(); ++i) {
            bitset_data.push_back(vector2bitset(data[i]));
        }
        fit(bitset_data, initialCentroidIndexs);
    }

    void fit(const std::vector<std::bitset<N>> &data,
             std::vector<unsigned int> initialCentroidIndexs = std::vector<unsigned int>()) {
        InitialzeCentroids(data, k_, this->init_center_type_, initialCentroidIndexs);
        for (unsigned int i = 0; i < data.size(); i++) this->assignments.push_back(0);

        // update hash tables
        for (unsigned int i = 0; i < k_; i++) {
            auto subvecs = SplitToSubSpace(this->centroids.at(i));
            for (unsigned int j = 0; j < subvecs.size(); j++) {
                this->tables_.at(j)[subvecs.at(j).to_ulong()].push_back(i);
            }
        }

        // select faster FindNN
        if (find_nn_type_ == BKmeansUtil::FindNNType::Auto) {
            find_nn_type_ = SelectFasterFindNNType(data);
        }

        // update centers
        long last_time = 0;
        for (unsigned int i = 0; i < iteration_; i++) {
            auto start = std::chrono::system_clock::now();
            this->UpdateCenter(data);
            auto end = std::chrono::system_clock::now();

            // record time & assignment filename
            if (verbose_)std::cout << "iteration" << i << "," << last_time << std::endl;
            last_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
    }


    void UpdateCenter(const std::vector<std::bitset<N>> &data) {
        this->error_ = 0;
        // assign and count
        std::vector<long> all_count;
        std::vector<std::vector<long>> count;
        for (unsigned int i = 0; i < this->k_; i++) {
            all_count.push_back(0);
            count.push_back(std::vector<long>(N));
        }

#pragma omp parallel for
        for (unsigned int i = 0; i < data.size(); i++) {
            assignments.at(i) = FindNearestCentroid(data.at(i));
        }
        // critical section
        for (unsigned int i = 0; i < data.size(); i++) {
            all_count[assignments[i]] += 1;
            for (unsigned int d = 0; d < N; d++) {
                if (data.at(i)[d] == 1) {
                    count.at(assignments.at(i)).at(d) += 1;
                } else {
                    count.at(assignments.at(i)).at(d) += -1;
                }
            }
        }
        if (verbose_)std::cout << "error:" << this->error_ << std::endl;

        // update
        for (unsigned int i = 0; i < this->k_; i++) {
            for (unsigned int d = 0; d < N; d++) {
                if (count.at(i).at(d) >= 0) {
                    centroids.at(i)[d] = 1;
                } else {
                    centroids.at(i)[d] = 0;
                }
            }
        }
    }

    int FindNearestCentroid(const std::vector<unsigned int> &query) {
        return FindNearestCentroid(vector2bitset(query));
    }

    int FindNearestCentroid(const std::bitset<N> &query) {
        if (find_nn_type_ == BKmeansUtil::FindNNType::Table) {
            return FindNNTable(query);
        } else if (find_nn_type_ == BKmeansUtil::FindNNType::Linear) {
            return FindNNLinear(query);
        } else {
            std::cerr << "ERROR: FINDNNTYPE" << std::endl;
            throw;
        }
    }

private:
    std::vector<std::vector<std::vector<int>>> tables_;
    std::vector<std::bitset<N>> centroids;
    std::vector<int> assignments;
    unsigned int k_;
    unsigned int iteration_;
    bool verbose_;
    BKmeansUtil::InitCenterType init_center_type_;
    unsigned long error_;
    unsigned long num_subspace_; //ceil(N/SUB)
    // [000] -> [100, 010, 001] -> [110, 101, ...]
    std::vector<std::vector<unsigned long>> bit_combinations_;
    std::vector<std::bitset<N>> bit_count_map_;

    BKmeansUtil::FindNNType SelectFasterFindNNType(const std::vector<std::bitset<N>> &data) {

        if (verbose_)std::cout << "Start SelectFasterFindNNType" << std::endl;
        unsigned int SAMPLE = 100;
        std::mt19937 mt(123);

        std::vector<std::bitset<N> > sampled_codes;
        for (unsigned int i = 0; i < SAMPLE; ++i) { // 100 samples
            int random_id = (unsigned int) mt() % (int) data.size();
            sampled_codes.push_back(data[random_id]);
        }

        // LINEAR
        auto start_linear = std::chrono::system_clock::now();
        for (const auto &code : sampled_codes) {
            FindNNLinear(code);
        }
        auto end_linear = std::chrono::system_clock::now();

        // TABLE
        auto start_table = std::chrono::system_clock::now();
        for (const auto &code : sampled_codes) {
            FindNNTable(code);
        }
        auto end_table = std::chrono::system_clock::now();

        // select faster method
        auto time_linear = std::chrono::duration_cast<std::chrono::nanoseconds>(end_linear - start_linear).count();
        auto time_table = std::chrono::duration_cast<std::chrono::nanoseconds>(end_table - start_table).count();
        std::cout << "<" << SAMPLE << "sample test> " <<
        "Linear: " << time_linear << "[ms]" <<
        "Table: " << time_table << "[ms]" << std::endl;
        if (time_linear < time_table) {
            if (verbose_)std::cout << "Use Linear" << std::endl;
            return BKmeansUtil::FindNNType::Linear;
        } else {
            if (verbose_)std::cout << "Use Table" << std::endl;
            return BKmeansUtil::FindNNType::Table;
        }
    }

    std::vector<std::bitset<SUB>> SplitToSubSpace(const std::bitset<N> &vec) {
        std::vector<std::bitset<SUB>> subvecs;
        for (unsigned int i = 0; i < vec.size(); i += SUB) {
            std::bitset<SUB> subvec = SliceBitSet(vec, i, i + SUB);
            subvecs.push_back(subvec);
        }
        return subvecs;
    }

    std::bitset<SUB> SliceBitSet(const std::bitset<N> &vec, unsigned int start, unsigned int end) {
        std::bitset<SUB> sub;
        for (unsigned int i = start; i < end; i++) {
            sub[i - start] = vec[i];
        }
        return sub;
    }

    std::vector<std::vector<unsigned long >> BitCombinations(size_t num_bits) {
        std::vector<std::vector<unsigned long>> ret;
        for (unsigned int target_bit = 0; target_bit < num_bits + 1; target_bit++) {
            std::vector<unsigned long> combinations;
            for (unsigned long num = 0; num < (unsigned long) (1 << num_bits); num++) {
                if (BitCount(num, num_bits) == target_bit) combinations.push_back(num);
            }
            ret.push_back(combinations);
        }
        return ret;
    }

    unsigned int BitCount(unsigned long value, unsigned int num_bits) {
        unsigned int count = 0;
        for (unsigned long mask = 1; mask < (unsigned long) (1 << num_bits); mask <<= 1) {
            if ((value & mask) != 0) count += 1;
        }
        return count;
    }

    unsigned int BitCount(std::bitset<N> value) {
        return value.count();
    }

    void InitialzeCentroids(const std::vector<std::bitset<N>> &data, unsigned int k,
                            BKmeansUtil::InitCenterType initCenterType,
                            std::vector<unsigned int> initialCentroidIndexs) {
        this->centroids.clear();
        std::mt19937 mt(0);
        if (initCenterType == BKmeansUtil::InitCenterType::Random) {
            std::uniform_int_distribution<unsigned long> randbit_generator(0, 1);
            // initialize centroids
            for (unsigned int i = 0; i < k; i++) {
                std::bitset<N> centroid;
                for (unsigned long j = 0; j < centroid.size(); j++) {
                    centroid[j] = randbit_generator(mt);
                }
                centroids.push_back(centroid);
            }
        } else if (initCenterType == BKmeansUtil::InitCenterType::RandomPick) {
            // initialize centroids with data
            std::uniform_int_distribution<unsigned long> randdataindex(0, data.size() - 1);
            for (unsigned int i = 0; i < k; i++) {
                std::bitset<N> copy(data.at(randdataindex(mt)));
                centroids.push_back(copy);
            }
        } else if (initCenterType == BKmeansUtil::InitCenterType::Outer) {
            for (auto &&index: initialCentroidIndexs) {
                std::bitset<N> copy(data.at(index));
                centroids.push_back(copy);
            }
        }
    }

    int FindNNTable(const std::bitset<N> &query) {
        auto subvecs = SplitToSubSpace(query);
        for (unsigned int subradius = 0; subradius < N; subradius++) {
            const auto differences = this->bit_combinations_.at(subradius);

            // is there any candidate really within radius from query?
            int minindex = -1;
            unsigned long mindistance = N;
            unsigned long cnt = 0;
            for (auto difference: differences) {
                for (unsigned int subindex = 0; subindex < this->num_subspace_; subindex++) {
                    for (auto &&candidate: this->tables_[subindex][subvecs[subindex].to_ulong() ^ difference]) {
                        cnt += 1;
                        auto distance = CalcDistance(this->centroids.at(candidate), query);
                        if (distance < mindistance &&
                            distance <= (subradius + 1) * this->num_subspace_ - 1) { // true_radius
                            minindex = candidate;
                            mindistance = distance;
                        }
                    }
                }
            }

            if (minindex != -1) {
                this->error_ += mindistance;
                return minindex;
            }
        }
        return -1;
    }

    int FindNNLinear(const std::bitset<N> &query) {
        int minindex = -1;
        unsigned long mindistance = N;
        for (unsigned int i = 0; i < this->k_; i++) {
            auto distance = CalcDistance(centroids.at(i), query);
            if (distance < mindistance) {
                minindex = i;
                mindistance = distance;
            }
        }
        this->error_ += mindistance;
        return minindex;
    }

    unsigned int CalcDistance(const std::bitset<N> &a, const std::bitset<N> &b) {
        return BitCount(a ^ b);
    }


};
}


#endif //PQKMEANS_BKMEANS_INTERNAL_H
