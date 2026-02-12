#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stelline/context.hh>

namespace py = pybind11;
using namespace stelline;

PYBIND11_MODULE(_manifest, m) {
    m.doc() = "Stelline manifest and metrics provider module";

    // Manifest providers.

    py::class_<ManifestProvider, std::shared_ptr<ManifestProvider>>(m, "ManifestProvider")
        .def(py::init<>());

    // Metrics providers.

    py::class_<MetricsProvider, std::shared_ptr<MetricsProvider>>(m, "MetricsProvider")
        .def(py::init<>())
        .def("push", &MetricsProvider::push)
        .def("collect", &MetricsProvider::collect);
}
