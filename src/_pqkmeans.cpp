#include <pybind11/pybind11.h>

#include "encoder/encoder_sample.h"
#include "clustering/cpp_implemented_clustering_sample.h"
#include "clustering/bkmeans.h"
#include "clustering/pqkmeans.h"

namespace py = pybind11;

namespace pqkmeans {
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
            .def(py::init <unsigned int, unsigned int, unsigned int > ())
            .def("fit", &BKMeans::fit)
            .def("predict_one", &BKMeans::predict_one)
            .def_property_readonly("labels_", &BKMeans::GetAssignments);

    py::class_<PQKmeans>(m, "PQKMeans")
            .def(py::init< std::vector<std::vector<std::vector<float>>>, int, int >())
            .def("fit", &PQKmeans::fit)
            .def("predict_one", &PQKmeans::predict_one);
}
}  // namespace pqkmeans
