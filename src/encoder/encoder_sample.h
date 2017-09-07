#ifndef PROJECT_ENCODER_SAMPLE_H
#define PROJECT_ENCODER_SAMPLE_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace pqkmeans {
    class EncoderSample {
    private:
        std::vector<std::vector<float>> index_dict;
    public:
        void fit_generator(py::iterator iterator);

        std::vector<long> transform_one(const std::vector<float> &vector);

        std::vector<float> inverse_transform_one(const std::vector<long> &vector);
    };
}

#endif //PROJECT_ENCODER_SAMPLE_H
