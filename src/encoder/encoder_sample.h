#ifndef PROJECT_ENCODER_SAMPLE_H
#define PROJECT_ENCODER_SAMPLE_H


#include <boost/python.hpp>
#include <vector>

class EncoderSample {
private:
    std::vector<boost::python::list> index_dict;
public:
    void fit_generator(boost::python::object iterator);

    boost::python::list transform_one(boost::python::list vector);

    boost::python::list inverse_transform_one(boost::python::list vector);
};

#endif //PROJECT_ENCODER_SAMPLE_H
