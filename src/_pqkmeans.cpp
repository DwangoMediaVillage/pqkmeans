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
            .def(py::init <unsigned int, unsigned int, unsigned int, unsigned int, bool > ())
            .def("fit", &BKMeans::fit)
            .def("predict_one", &BKMeans::predict_one)
            .def_property_readonly("labels_", &BKMeans::GetAssignments)
            .def_property_readonly("cluster_centers_", &BKMeans::GetClusterCenters);

    py::class_<PQKMeans>(m, "PQKMeans")
            .def(py::init< std::vector<std::vector<std::vector<float>>>, int, int, bool, std::vector<std::vector<unsigned char>>>())
            .def("fit", &PQKMeans::fit)
            .def("predict_one", &PQKMeans::predict_one)
            .def("set_cluster_centers", &PQKMeans::SetClusterCenters)
            .def_property_readonly("iteration_", &PQKMeans::Iteration)
            .def_property_readonly("k_", &PQKMeans::K)
            .def_property_readonly("labels_", &PQKMeans::GetAssignments)
            .def_property_readonly("cluster_centers_", &PQKMeans::GetClusterCenters);

}
}  // namespace pqkmeans
