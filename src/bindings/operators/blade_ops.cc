#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/blade/base.hh>
#include <stelline/yaml/types/block_shape.hh>
#include <stelline/yaml/types/map.hh>

namespace py = pybind11;
using namespace stelline::operators::blade;
using namespace holoscan;

class PyCorrelatorOp : public CorrelatorOp {
public:
    using CorrelatorOp::CorrelatorOp;

    PyCorrelatorOp(Fragment* fragment,
                   const py::args& args,
                   uint64_t number_of_buffers,
                   const stelline::BlockShape& input_shape,
                   const stelline::BlockShape& output_shape,
                   const stelline::Map& options,
                   const std::string& name = "correlator")
        : CorrelatorOp(ArgList{Arg("number_of_buffers", number_of_buffers),
                                Arg("input_shape", input_shape),
                                Arg("output_shape", output_shape),
                                Arg("options", options)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyBeamformerOp : public BeamformerOp {
public:
    using BeamformerOp::BeamformerOp;

    PyBeamformerOp(Fragment* fragment,
                   const py::args& args,
                   uint64_t number_of_buffers,
                   const stelline::BlockShape& input_shape,
                   const stelline::BlockShape& output_shape,
                   const stelline::Map& options,
                   const std::string& name = "beamformer")
        : BeamformerOp(ArgList{Arg("number_of_buffers", number_of_buffers),
                               Arg("input_shape", input_shape),
                               Arg("output_shape", output_shape),
                               Arg("options", options)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyFrbnnOp : public FrbnnOp {
public:
    using FrbnnOp::FrbnnOp;

    PyFrbnnOp(Fragment* fragment,
              const py::args& args,
              uint64_t number_of_buffers,
              const stelline::BlockShape& input_shape,
              const stelline::BlockShape& output_shape,
              const stelline::Map& options,
              const std::string& name = "frbnn")
        : FrbnnOp(ArgList{Arg("number_of_buffers", number_of_buffers),
                          Arg("input_shape", input_shape),
                          Arg("output_shape", output_shape),
                          Arg("options", options)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_blade_ops, m) {
    m.doc() = "Stelline blade operators module";

    py::class_<CorrelatorOp, PyCorrelatorOp, Operator, std::shared_ptr<CorrelatorOp>>(m, "CorrelatorOp")
        .def(py::init<Fragment*, const py::args&, uint64_t, const stelline::BlockShape&,
                      const stelline::BlockShape&, const stelline::Map&, const std::string&>(),
             py::arg("fragment"),
             py::arg("number_of_buffers"),
             py::arg("input_shape"),
             py::arg("output_shape"),
             py::arg("options"),
             py::arg("name") = "correlator")
        .def("collect_metrics_map", &CorrelatorOp::collectMetricsMap)
        .def("collect_metrics_string", &CorrelatorOp::collectMetricsString);

    py::class_<BeamformerOp, PyBeamformerOp, Operator, std::shared_ptr<BeamformerOp>>(m, "BeamformerOp")
        .def(py::init<Fragment*, const py::args&, uint64_t, const stelline::BlockShape&,
                      const stelline::BlockShape&, const stelline::Map&, const std::string&>(),
             py::arg("fragment"),
             py::arg("number_of_buffers"),
             py::arg("input_shape"),
             py::arg("output_shape"),
             py::arg("options"),
             py::arg("name") = "beamformer")
        .def("collect_metrics_map", &BeamformerOp::collectMetricsMap)
        .def("collect_metrics_string", &BeamformerOp::collectMetricsString);

    py::class_<FrbnnOp, PyFrbnnOp, Operator, std::shared_ptr<FrbnnOp>>(m, "FrbnnOp")
        .def(py::init<Fragment*, const py::args&, uint64_t, const stelline::BlockShape&,
                      const stelline::BlockShape&, const stelline::Map&, const std::string&>(),
             py::arg("fragment"),
             py::arg("number_of_buffers"),
             py::arg("input_shape"),
             py::arg("output_shape"),
             py::arg("options"),
             py::arg("name") = "frbnn")
        .def("collect_metrics_map", &FrbnnOp::collectMetricsMap)
        .def("collect_metrics_string", &FrbnnOp::collectMetricsString);
}
