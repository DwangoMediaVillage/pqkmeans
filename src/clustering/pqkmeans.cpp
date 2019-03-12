#include "./pqkmeans.h"

namespace pqkmeans {


  

PQKMeans::PQKMeans(std::vector<std::vector<std::vector<float> > > codewords, int K, int itr, bool verbose)
    : codewords_(codewords), K_(K), iteration_(itr), verbose_(verbose)
{
    assert(!codewords.empty() && !codewords[0].empty() && !codewords[0][0].empty());
    M_ = codewords.size(); // The number of subspace
    std::size_t Ks = codewords[0].size();  // The number of codewords for each subspace

    if (256 < Ks) {
        std::cerr << "Error. Ks is too large. "
                  << "Currently, we only support PQ code with Ks <= 256 "
                  << "so that each subspace is represented by unsigned char (8 bit)"
                  << std::endl;
        throw;
    }

    // Compute distance-matrices among codewords
    distance_matrices_among_codewords_.resize(
                M_, std::vector<std::vector<float>>(Ks, std::vector<float>(Ks, 0)));

    for (std::size_t m = 0; m < M_; ++m) {
        for (std::size_t k1 = 0; k1 < Ks; ++k1) {
            for (std::size_t k2 = 0; k2 < Ks; ++k2) {
                distance_matrices_among_codewords_[m][k1][k2] =
                        L2SquaredDistance(codewords[m][k1], codewords[m][k2]);
            }
        }
    }
}

int PQKMeans::predict_one(const std::vector<unsigned char> &pyvector)
{
    assert(pyvector.size() == M_);
    std::pair<std::size_t, float> nearest_one = FindNearetCenterLinear(pyvector, centers_);
    return (int) nearest_one.first;
}



void PQKMeans::fit(const std::vector<unsigned char> &pydata) {
    assert( (size_t) K_ * M_ <= pydata.size());
    assert(pydata.size() % M_ == 0);
    std::size_t N = pydata.size() / M_;

    // Refresh
    centers_.clear();
    centers_.resize((std::size_t) K_, std::vector<unsigned char>(M_));
    assignments_.clear();
    assignments_.resize(N);
    assignments_.shrink_to_fit(); // If the previous fit malloced a long assignment array, shrink it.

    // Prepare data temporal buffer
    std::vector<std::vector<unsigned char>> centers_new, centers_old;

    // (1) Initialization
    // [todo] Currently, only random pick is supported
    InitializeCentersByRandomPicking(pydata, K_, &centers_new);

    // selected_indices_foreach_center[k] has indices, where
    // each pydata[id] is assigned to k-th center.
    std::vector<std::vector<std::size_t>> selected_indices_foreach_center(K_);
    for (auto &selected_indices : selected_indices_foreach_center) {
        selected_indices.reserve( N / K_); // roughly allocate
    }

    std::vector<double> errors(N, 0);

    for (int itr = 0; itr < iteration_; ++itr) {
        if (verbose_) {
            std::cout << "Iteration start: " << itr << " / " << iteration_ << std::endl;
        }
        auto start = std::chrono::system_clock::now(); // ---- timer start ---

        centers_old = centers_new;

        // (2) Find NN centers
        selected_indices_foreach_center.clear();
        selected_indices_foreach_center.resize(K_);

        double error_sum = 0;

#pragma omp parallel for
        for(long long n_tmp = 0LL; n_tmp < static_cast<long long>(N); ++n_tmp) {
            std::size_t n = static_cast<std::size_t>(n_tmp);
            std::pair<std::size_t, float> min_k_dist = FindNearetCenterLinear(NthCode(pydata, n), centers_old);
            assignments_[n] = (int) min_k_dist.first;
            errors[n] = min_k_dist.second;
        }
        // (2.5) assignments -> selected_indices_foreach_center
        for (std::size_t n = 0; n < N; ++n) {
            int k = assignments_[n];
            selected_indices_foreach_center[k].push_back(n);
            error_sum += errors[n];
        }

        if (verbose_) {
            std::cout << "find_nn finished. Error: " << error_sum / N << std::endl;
            std::cout << "find_nn_time,"
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
                      << std::endl;
        }

        // (3) Compute centers
        if (itr != iteration_ - 1) {
            // Usually, centers would be updated.
            // After the last assignment, centers should not be updated, so this block is skiped.

            for (int k = 0; k < K_; ++k) {
                if (selected_indices_foreach_center[k].empty()) {
                    if (verbose_) {
                        std::cout << "Caution. No codes are assigned to " << k << "-th centers." << std::endl;
                    }
                    continue;
                }
                centers_new[k] =
                        ComputeCenterBySparseVoting(pydata, selected_indices_foreach_center[k]);
            }
        }
        if (verbose_) {
            std::cout << "find_nn+update_center_time,"
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
                      << std::endl;
        }

    }
    centers_ = centers_new;
}

const std::vector<int> PQKMeans::GetAssignments()
{
    return assignments_;
}

std::vector<std::vector<unsigned char>> PQKMeans::GetClusterCenters()
{
    return centers_;
}

void PQKMeans::SetClusterCenters(const std::vector<std::vector<unsigned char>> &centers_new)
{
    assert(centers_new.size() == (size_t) K_);
    centers_ = centers_new;
}


float PQKMeans::SymmetricDistance(const std::vector<unsigned char> &code1,
                                  const std::vector<unsigned char> &code2)
{
    // assert(code1.size() == code2.size());
    // assert(code1.size() == M_);
    float dist = 0;
    for (std::size_t m = 0; m < M_; ++m) {
        dist += distance_matrices_among_codewords_[m][code1[m]][code2[m]];
    }
    return dist;
}

float PQKMeans::L2SquaredDistance(const std::vector<float> &vec1,
                                  const std::vector<float> &vec2)
{
    assert(vec1.size() == vec2.size());
    float dist = 0;
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return dist;
}



void PQKMeans::InitializeCentersByRandomPicking(const std::vector<unsigned char> &codes, int K, std::vector<std::vector<unsigned char> > *centers)
{
    assert(centers != nullptr);
    centers->clear();
    centers->resize(K);

    std::vector<int> ids(codes.size() / M_);
    std::iota(ids.begin(), ids.end(), 0); // 0, 1, 2, ..., codes.size()-1
    std::mt19937 random_engine(0);
    std::shuffle(ids.begin(), ids.end(), random_engine);
    for (std::size_t k = 0; k < (std::size_t) K; ++k) {
        (*centers)[k] = NthCode(codes, ids[k]);
    }

}

std::pair<std::size_t, float> PQKMeans::FindNearetCenterLinear(const std::vector<unsigned char> &query,
                                                               const std::vector<std::vector<unsigned char> > &codes)
{
    std::vector<float> dists(codes.size());

    // Compute a distance from a query to each code in parallel
    long long sz = static_cast<long long>(codes.size());
#pragma omp parallel for
    for (long long i_tmp = 0; i_tmp < sz; ++i_tmp) {
        std::size_t i = static_cast<std::size_t>(i_tmp);
        dists[i] = SymmetricDistance(query, codes[i]);
    }

    // Just pick up the closest one
    float min_dist = FLT_MAX;
    int min_i = -1;
    for (size_t i = 0, sz = codes.size(); i < sz; ++i) {
        if (dists[i] < min_dist) {
            min_i = i;
            min_dist = dists[i];
        }
    }
    assert(min_i != -1);

    return std::pair<std::size_t, float>((std::size_t) min_i, min_dist);
}




std::vector<unsigned char> PQKMeans::ComputeCenterBySparseVoting(const std::vector<unsigned char> &codes, const std::vector<std::size_t> &selected_ids)
{
    std::vector<unsigned char> average_code(M_);
    std::size_t Ks = codewords_[0].size();  // The number of codewords for each subspace

    for (std::size_t m = 0; m < M_; ++m) {
        // Scan the assigned codes, then create a freq-histogram
        std::vector<int> frequency_histogram(Ks, 0);
        for (const auto &id : selected_ids) {
            ++frequency_histogram[ NthCodeMthElement(codes, id, m) ];
        }

        // Vote the freq-histo with weighted by ditance matrices
        std::vector<float> vote(Ks, 0);
        for (std::size_t k1 = 0; k1 < Ks; ++k1) {
            int freq = frequency_histogram[k1];
            if (freq == 0) { // not assigned for k1. Skip it.
                continue;
            }
            for (std::size_t k2 = 0; k2 < Ks; ++k2) {
                vote[k2] += (float) freq * distance_matrices_among_codewords_[m][k1][k2];
            }
        }

        // find min
        float min_dist = FLT_MAX;
        int min_ks = -1;
        for (std::size_t ks = 0; ks < Ks; ++ks) {
            if (vote[ks] < min_dist) {
                min_ks = (int) ks;
                min_dist = vote[ks];
            }
        }
        assert(min_ks != -1);
        average_code[m] = (unsigned char) min_ks;
    }
    return average_code;
}

std::vector<unsigned char> PQKMeans::NthCode(const std::vector<unsigned char> &long_code, std::size_t n)
{
    return std::vector<unsigned char>(long_code.begin() + n * M_, long_code.begin() + (n + 1) * M_);
}

unsigned char PQKMeans::NthCodeMthElement(const std::vector<unsigned char> &long_code, std::size_t n, int m)
{
    return long_code[ n * M_ + m];
}



} // namespace pqkmeans
