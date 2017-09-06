#include <pybind11/pybind11.h>

#include "encoder/encoder_sample.h"
#include "clustering/cpp_implemented_clustering_sample.h"
#include "clustering/bkmeans.h"

namespace py = pybind11;


PYBIND11_MODULE(_pqkmeans, m) {
    py::class_<EncoderSample>(m, "EncoderSample")
            .def(py::init<>())
            .def("fit_generator", &EncoderSample::fit_generator)
            .def("transform_one", &EncoderSample::transform_one)
            .def("inverse_transform_one", &EncoderSample::inverse_transform_one);

    py::class_<CppImplementedClusteringSample>(m, "CppImplementedClusteringSample")
            .def(py::init<>())
            .def("fit_one", &CppImplementedClusteringSample::fit_one)
            .def("predict_one", &CppImplementedClusteringSample::predict_one);

    py::class_<BKMeans>(m, "BKMeans")
            .def(py::init<unsigned int, unsigned int>())
            .def("fit_one", &BKMeans::fit_one)
            .def("predict_one", &BKMeans::predict_one);

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}