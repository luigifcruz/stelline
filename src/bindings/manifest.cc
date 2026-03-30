#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stelline/context.hh>

namespace py = pybind11;
using namespace stelline;

PYBIND11_MODULE(_manifest, m) {
    m.doc() = "Stelline manifest and metrics provider module";

    // Manifest providers.

    py::class_<ManifestProvider, std::shared_ptr<ManifestProvider>>(m, "ManifestProvider")
        .def(py::init<>())
        .def("store", [](ManifestProvider& self, const std::string& key, py::object value,
                         const std::string& type, uint64_t start, uint64_t end) {
            std::any val;
            if (type == "f64") {
                val = value.cast<double>();
            } else if (type == "f32") {
                val = value.cast<float>();
            } else if (type == "u64") {
                val = value.cast<uint64_t>();
            } else if (type == "i64") {
                val = value.cast<int64_t>();
            } else if (type == "i32") {
                val = static_cast<int32_t>(value.cast<int64_t>());
            } else if (type == "string") {
                val = value.cast<std::string>();
            } else {
                val = value;
            }
            self.store(key, val, start, end);
        }, py::arg("key"), py::arg("value"), py::arg("type") = "string",
           py::arg("start") = 0, py::arg("end") = UINT64_MAX)
        .def("fetch", [](ManifestProvider& self, const std::string& key, uint64_t timestamp) -> py::object {
            std::any result;
            try {
                self.fetch(key, result, timestamp);
            } catch (const std::runtime_error&) {
                return py::none();
            }
            if (auto* v = std::any_cast<double>(&result)) {
                return py::cast(*v);
            } else if (auto* v = std::any_cast<float>(&result)) {
                return py::cast(*v);
            } else if (auto* v = std::any_cast<int64_t>(&result)) {
                return py::cast(*v);
            } else if (auto* v = std::any_cast<uint64_t>(&result)) {
                return py::cast(*v);
            } else if (auto* v = std::any_cast<int32_t>(&result)) {
                return py::cast(*v);
            } else if (auto* v = std::any_cast<std::string>(&result)) {
                return py::cast(*v);
            } else if (auto* v = std::any_cast<py::object>(&result)) {
                return *v;
            }
            return py::none();
        });

    // Metric struct.

    py::class_<Metric>(m, "Metric")
        .def_readwrite("type", &Metric::type)
        .def_readwrite("value", &Metric::value);

    // Metrics providers.

    py::class_<MetricsProvider, std::shared_ptr<MetricsProvider>>(m, "MetricsProvider")
        .def(py::init<>())
        .def("record", &MetricsProvider::record, py::arg("key"), py::arg("value"),
             py::arg("type") = "text", py::arg("global") = false)
        .def("snapshot", &MetricsProvider::snapshot, py::arg("global") = false);
}
