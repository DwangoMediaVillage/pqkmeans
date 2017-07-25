#include "encoder_sample.h"

void EncoderSample::fit_generator(boost::python::object iterator) {
    boost::python::stl_input_iterator<boost::python::list> begin(iterator), end;
    for (auto itr = begin; itr != end; ++itr) {
        auto item = *itr;
        std::vector<boost::python::object>::iterator found = std::find(index_dict.begin(), index_dict.end(), item);
        if (found == index_dict.end()) {
            index_dict.push_back(item);
        }
    }
}


boost::python::list EncoderSample::transform_one(boost::python::list vector) {
    std::vector<boost::python::list>::iterator found = std::find(index_dict.begin(), index_dict.end(), vector);
    if (found != index_dict.end()) {
        boost::python::list ret_vector;
        ret_vector.append(std::distance(index_dict.begin(), found));
        return ret_vector;
    } else {
        throw std::invalid_argument("couldn't handle input vector");
    }
}

boost::python::list EncoderSample::inverse_transform_one(boost::python::list vector) {
    long value = boost::python::extract<long>(vector[0]);
    std::vector<boost::python::object>::iterator found = std::find(index_dict.begin(), index_dict.end(), vector);
    if (value < index_dict.size()) {
        return index_dict[value];
    } else {
        throw std::invalid_argument("couldn't handle input vector");
    }
}
