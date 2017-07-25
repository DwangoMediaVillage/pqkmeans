#ifndef PROJECT_ENCODER_SAMPLE_H
#define PROJECT_ENCODER_SAMPLE_H


#include <boost/python.hpp>
#include <vector>

class EncoderSample {
private:
    std::vector<boost::python::object> index_dict;
public:
    void fit_generator(boost::python::object iterator);

    boost::python::list transform_one(boost::python::list vector);

    void inverse_transform(boost::python::object iterator);

};


#endif //PROJECT_ENCODER_SAMPLE_H
