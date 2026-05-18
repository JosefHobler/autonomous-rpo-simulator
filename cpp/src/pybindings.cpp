#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "rpo/ekf_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ekf_cpp, m) {
    m.doc() = "C++ EKF core (Eigen) exposed via pybind11.";

    py::class_<rpo::EkfCore>(m, "EkfCore")
        .def(py::init<int>(), py::arg("n_state"))
        .def("set_state", &rpo::EkfCore::set_state, py::arg("x"), py::arg("P"))
        .def("state",      &rpo::EkfCore::state,      py::return_value_policy::reference_internal)
        .def("covariance", &rpo::EkfCore::covariance, py::return_value_policy::reference_internal)
        .def("predict",       &rpo::EkfCore::predict,       py::arg("F"), py::arg("Q"))
        .def("update_linear", &rpo::EkfCore::update_linear, py::arg("y"), py::arg("H"), py::arg("R"))
        .def("dim", &rpo::EkfCore::dim);
}
