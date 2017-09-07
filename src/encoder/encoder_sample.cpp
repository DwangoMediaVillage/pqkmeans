#include "encoder_sample.h"

namespace pqkmeans {
void EncoderSample::fit_generator(py::iterator iterator) {
    auto begin = std::begin(iterator);
    auto end = std::end(iterator);
    for (auto itr = begin; itr != end; ++itr) {
        auto item = (*itr).cast<std::vector<float>>();
        auto found = std::find(index_dict.begin(), index_dict.end(), item);

        if (found == index_dict.end()) {
            index_dict.push_back(item);
        }
    }
}

std::vector<long> EncoderSample::transform_one(const std::vector<float> &vector) {
    auto found = std::find(index_dict.begin(), index_dict.end(), vector);
    if (found != index_dict.end()) {
        std::vector<long> ret_vector;
        ret_vector.push_back(std::distance(index_dict.begin(), found));
        return ret_vector;
    } else {
        throw std::invalid_argument("couldn't handle input vector");
    }
}

std::vector<float> EncoderSample::inverse_transform_one(const std::vector<long> &vector) {
    std::size_t value = (std::size_t) vector[0];
    if (value < index_dict.size()) {
        return index_dict[value];
    } else {
        throw std::invalid_argument("couldn't handle input vector");
    }
}
}
