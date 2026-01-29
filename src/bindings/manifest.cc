#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stelline/manifest.hh>

namespace py = pybind11;
using namespace stelline;

PYBIND11_MODULE(_manifest, m) {
    m.doc() = "Stelline manifest module";

    py::class_<ManifestProvider, std::shared_ptr<ManifestProvider>>(m, "ManifestProvider")
        .def(py::init<const std::string&>(), py::arg("endpoint"));
}
