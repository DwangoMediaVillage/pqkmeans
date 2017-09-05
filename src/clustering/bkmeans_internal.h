//

#ifndef PQKMEANS_BKMEANS_IMPL_H
#define PQKMEANS_BKMEANS_IMPL_H

#include <iostream>
#include <sstream>
#include <random>
#include <climits>
#include "i_bkmeans_internal.h"

namespace BKmeansUtil {
    enum InitCenterType {
        RandomPick, Random, Outer
    };
    enum FindNNType {
        Table, Linear, AUTO
    };
}


template<size_t N, size_t SUB>
class BKmeansInternal: public IBKmeansInternal {
public:
    std::vector<std::bitset<N>> centroids;
    std::vector<unsigned int> assignments;
    BKmeansUtil::FindNNType findNNType;

    BKmeansInternal(const std::vector<std::bitset<N>> &data, unsigned int k, unsigned int subspace, unsigned int iteration,
                bool store_assignment, const char *assignments_dir,
                BKmeansUtil::InitCenterType initCenterType = BKmeansUtil::InitCenterType::RandomPick,
                std::vector<unsigned int> initialCentroidIndexs = std::vector<unsigned int>()) :
            findNNType(BKmeansUtil::FindNNType::AUTO) {

        this->k = k;
        this->subspace = subspace;

        std::cout << "init center" << std::endl;
        InitialzeCentroids(data, k, initCenterType, initialCentroidIndexs);

        for (unsigned int i = 0; i < data.size(); i++) this->assignments.push_back(0);

        // initialize hash tables
        std::cout << "init table" << std::endl;
        for (unsigned int i = 0; i < N; i += this->subspace) {
            std::cout << "init table: " << i << ": " << (1UL << this->subspace) << std::endl;
////            std::vector<std::vector<int>> table(1UL << this->subspace);
//            std::vector<std::vector<int>> table(1UL << (this->subspace-1));
            std::vector<unsigned long> table(1UL << this->subspace);
            std::cout << "init table: " << i << ": " << table.size() << std::endl;
            this->tables.push_back(table);
        }
        this->num_subspace = this->tables.size();

        // update hash tables
        std::cout << "update table" << std::endl;
        for (unsigned int i = 0; i < k; i++) {
            std::cout << this->centroids.at(i) << std::endl;
            auto subvecs = SplitToSubSpace(this->centroids.at(i));
            std::cout << subvecs[0].to_ulong() << std::endl;
            std::cout << "subvec: " << subvecs.size() << std::endl;
            for (unsigned int j = 0; j < subvecs.size(); j++) {
                this->tables.at(j)[subvecs.at(j).to_ulong()] = i;
            }
        }
        for (unsigned int i = 0; i < N; i++) {
            std::bitset<N> bc;
            bc[i] = 1;
            this->bit_count_map.push_back(bc);
        }
        this->bit_combinations = BitCombinations(this->subspace);

        // select faster FindNN
        if (findNNType == BKmeansUtil::FindNNType::AUTO) {
            findNNType = SelectFasterFindNNType(data);
        }

        // update centers
        long last_time = 0;
        for (unsigned int i = 0; i < iteration; i++) {
            auto start = std::chrono::system_clock::now();
            this->UpdateCenter(data);
            auto end = std::chrono::system_clock::now();

            std::stringstream file_name;
            file_name << assignments_dir << "/assignment_" << i << ".dat";
//            ProjUtil::CsvData::WriteVector<unsigned int>(file_name.str().c_str(), this->assignments);

            // record time & assignment filename
            std::cout << "iteration" << i << "," << last_time << "," << file_name.str() << std::endl;
            last_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
    }

    void UpdateCenter(const std::vector<std::bitset<N>> &data) {
        this->error = 0;
        // assign and count
        std::vector<long> all_count;
        std::vector<std::vector<long>> count;
        for (unsigned int i = 0; i < this->k; i++) {
            all_count.push_back(0);
            count.push_back(std::vector<long>(N));
        }

#pragma omp parallel for
        for (unsigned int i = 0; i < data.size(); i++) {
            if (findNNType == BKmeansUtil::FindNNType::Table) {
                assignments.at(i) = FindNN(data.at(i));
            } else if (findNNType == BKmeansUtil::FindNNType::Linear) {
                assignments.at(i) = FindNNLinear(data.at(i));
            } else {
                std::cerr << "ERROR: FINDNNTYPE" << std::endl;
                throw;
            }
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
        std::cout << "error:" << this->error << std::endl;

        // debug
        for (auto &&cnt: count[10]) {
            if (cnt == all_count[assignments[10]]) {
                std::cout << cnt << ",";
            }
        }
        std::cout << std::endl;

        // update
        for (unsigned int i = 0; i < this->k; i++) {
            for (unsigned int d = 0; d < N; d++) {
                if (count.at(i).at(d) >= 0) {
                    centroids.at(i)[d] = 1;
                } else {
                    centroids.at(i)[d] = 0;
                }
            }
        }
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

    unsigned int FindNN(const std::bitset<N> &query) {
        auto subvecs = SplitToSubSpace(query);
        for (unsigned int subradius = 0; subradius < N; subradius++) {
            std::cout << "subradius: " << subradius << std::endl;
            std::vector<unsigned long> &differences = this->bit_combinations.at(subradius);
////            std::unordered_set<int> candidates;
//            std::vector<int> candidates;
////            candidates.reserve();
//            candidates.clear();

            // is there any candidate really within radius from query?
            unsigned int minindex = UINT_MAX;
            bool found_nn = false;
            unsigned long mindistance = N + 1;
            unsigned long cnt = 0;
            for (unsigned long difference: differences) {
                std::cout << "difference: " << difference << std::endl;
                for (unsigned int subindex = 0; subindex < this->num_subspace; subindex++) {
//                    auto candidate = (this->tables[0][subvecs[subindex].to_ulong() ^ difference]);
//                    unsigned long candidate = (this->tables.at(subindex).at((subvecs.at(subindex).to_ulong()) ^ difference));
                    unsigned long candidate = (this->tables.at(subindex).at(subvecs.at(subindex).to_ulong()));
                    cnt += 1;
//                    std::cout<<
                    auto distance = CalcDistance(this->centroids.at(candidate), query);
                    std::cout << "distance " << distance << std::endl;
                    std::cout << distance << "<" << mindistance << std::endl;
                    std::cout << (subradius + 1) << "*" << this->num_subspace - 1 << std::endl;
                    std::cout << distance << "<=" << (subradius + 1) * this->num_subspace - 1 << std::endl;
                    if (distance < mindistance &&
                        distance <= (subradius + 1) * this->num_subspace - 1) { // true_radius
                        std::cout << "true radius" << std::endl;
                        minindex = candidate;
                        found_nn = true;
                        mindistance = distance;
                    }
                }
            }
            std::cout << "foundnn: " << found_nn << std::endl;
            //std::cout<<"calc for "<<cnt<<"times"<<std::endl;

            if (found_nn) {
                this->error += mindistance;
                return minindex;
            }
        }
        throw "cannot find NN";
    }

    unsigned int FindNNLinear(const std::bitset<N> &query) {
        unsigned int minindex = UINT_MAX;
        bool found_nn = false;
        unsigned long mindistance = N;
        for (unsigned int i = 0; i < this->k; i++) {
            auto distance = CalcDistance(centroids.at(i), query);
            if (distance < mindistance) {
                minindex = i;
                mindistance = distance;
                found_nn = true;
            }
        }
//        std::cout<<mindistance<<std::endl;
        this->error += mindistance;
        if (found_nn) {
            return minindex;
        } else {
            throw "cannot find NN";
        }
    }

    unsigned int CalcDistance(const std::bitset<N> &a, const std::bitset<N> &b) {
        return (unsigned int) BitCount(a ^ b);
    }

private:
    std::vector<std::vector<unsigned long>> tables;
    unsigned int k;
    unsigned long error;
    unsigned int subspace;
    unsigned long num_subspace; //ceil(N/SUB)
    // [000] -> [100, 010, 001] -> [110, 101, ...]
    std::vector<std::vector<unsigned long>> bit_combinations;
    std::vector<std::bitset<N>> bit_count_map;

    BKmeansUtil::FindNNType SelectFasterFindNNType(const std::vector<std::bitset<N>> &data) {
        std::cout << "Start SelectFasterFindNNType" << std::endl;
        unsigned int SAMPLE = 100;
        std::mt19937 mt(123);


        std::vector<std::bitset<N>> sampled_codes;
        for (unsigned int i = 0; i < SAMPLE; ++i) { // 100 samples
            std::cout << "create sample" << std::endl;
            int random_id = (unsigned int) mt() % (int) data.size();
            sampled_codes.push_back(data[random_id]);
        }

        // LINEAR
        std::cout << "test linear" << std::endl;
        auto start_linear = std::chrono::system_clock::now();
        for (const auto &code : sampled_codes) {
            FindNNLinear(code);
        }
        auto end_linear = std::chrono::system_clock::now();

        // TABLE
        std::cout << "test table" << std::endl;
        auto start_table = std::chrono::system_clock::now();
        for (const auto &code : sampled_codes) {
            std::cout << "find nn" << std::endl;
            FindNN(code);
        }
        auto end_table = std::chrono::system_clock::now();

        // select faster method
        auto time_linear = std::chrono::duration_cast<std::chrono::nanoseconds>(end_linear - start_linear).count();
        auto time_table = std::chrono::duration_cast<std::chrono::nanoseconds>(end_table - start_table).count();
        std::cout << "<" << SAMPLE << "sample test> " <<
        "Linear: " << time_linear << "[ms]" <<
        "Table: " << time_table << "[ms]" << std::endl;
        if (time_linear < time_table) {
            std::cout << "Use Linear" << std::endl;
            return BKmeansUtil::FindNNType::Linear;
        } else {
            std::cout << "Use Table" << std::endl;
            return BKmeansUtil::FindNNType::Table;
        }
    }

    std::vector<std::bitset<SUB>> SplitToSubSpace(const std::bitset<N> &vec) {
        std::vector<std::bitset<SUB>> subvecs;
        for (unsigned int i = 0; i < vec.size(); i += subspace) {
            std::bitset<SUB> subvec = SliceBitSet(vec, i, i + subspace);
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

    std::vector<std::vector<unsigned long >> BitCombinations(unsigned int num_bits) {
        std::vector<std::vector<unsigned long>> ret;
        for (unsigned int target_bit = 0; target_bit < num_bits + 1; target_bit++) {
            std::vector<unsigned long> combinations;
            for (unsigned long num = 0; num < 1 << num_bits; num++) {
                if (BitCount(num, num_bits) == target_bit) combinations.push_back(num);
            }
            ret.push_back(combinations);
        }
        return ret;
    }

    unsigned int BitCount(unsigned long value, unsigned int num_bits) {
        unsigned int count = 0;
        for (unsigned long mask = 1; mask < 1 << num_bits; mask <<= 1) {
            if ((value & mask) != 0) count += 1;
        }
        return count;
    }

    size_t BitCount(std::bitset<N> value) {
//        unsigned int count = 0;
//        for(auto&& mask: bit_count_map) count+=(value&mask);
////        for(unsigned long i=0; i<N; i++){
////            if(value[i] == 1) count+=1;
////        }
        return value.count();
    }
};


#endif //PQKMEANS_BKMEANS_IMPL_H
