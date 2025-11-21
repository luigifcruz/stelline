#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/socket/base.hh>

namespace py = pybind11;
using namespace stelline::operators::socket;
using namespace holoscan;

class PyZmqTransmitterOp : public ZmqTransmitterOp {
public:
    using ZmqTransmitterOp::ZmqTransmitterOp;

    PyZmqTransmitterOp(Fragment* fragment,
                       const py::args& args,
                       const std::string& address,
                       const std::string& name = "zmq_transmitter")
        : ZmqTransmitterOp(ArgList{Arg("address", address)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_socket_ops, m) {
    m.doc() = "Stelline socket operators module";

    py::class_<ZmqTransmitterOp, PyZmqTransmitterOp, Operator, std::shared_ptr<ZmqTransmitterOp>>(m, "ZmqTransmitterOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("address"),
             py::arg("name") = "zmq_transmitter")
        .def("collect_metrics_map", &ZmqTransmitterOp::collectMetricsMap)
        .def("collect_metrics_string", &ZmqTransmitterOp::collectMetricsString);
}
