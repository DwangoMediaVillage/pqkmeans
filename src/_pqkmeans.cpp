#include <boost/python.hpp>

#include "encoder/encoder_sample.h"

BOOST_PYTHON_MODULE (_pqkmeans) {
    using namespace boost::python;
    class_<EncoderSample>("EncoderSample")
            .def("fit_generator", &EncoderSample::fit_generator)
            .def("transform_one", &EncoderSample::transform_one);
}