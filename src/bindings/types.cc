#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stelline/types.hh>
#include <stelline/yaml/types/block_shape.hh>

namespace py = pybind11;
using namespace stelline;

PYBIND11_MODULE(_types, m) {
    m.doc() = "Stelline types module";

    py::class_<BlockShape, std::shared_ptr<BlockShape>>(m, "BlockShape")
        .def(py::init<>())
        .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t>(),
             py::arg("number_of_antennas"),
             py::arg("number_of_channels"),
             py::arg("number_of_samples"),
             py::arg("number_of_polarizations"))
        .def_readwrite("number_of_antennas", &BlockShape::numberOfAntennas)
        .def_readwrite("number_of_channels", &BlockShape::numberOfChannels)
        .def_readwrite("number_of_samples", &BlockShape::numberOfSamples)
        .def_readwrite("number_of_polarizations", &BlockShape::numberOfPolarizations)
        .def("__repr__", [](const BlockShape& b) {
            return fmt::format("BlockShape(A={}, C={}, S={}, P={})",
                             b.numberOfAntennas, b.numberOfChannels,
                             b.numberOfSamples, b.numberOfPolarizations);
        });
}
