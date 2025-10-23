#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/transport/base.hh>
#include <stelline/yaml/types/block_shape.hh>

namespace py = pybind11;
using namespace stelline::operators::transport;
using namespace holoscan;

class PyAtaReceiverOp : public AtaReceiverOp {
public:
    using AtaReceiverOp::AtaReceiverOp;

    PyAtaReceiverOp(Fragment* fragment, const py::args& args,
                    const stelline::BlockShape& total_block,
                    const stelline::BlockShape& partial_block,
                    const stelline::BlockShape& offset_block,
                    uint64_t concurrent_blocks,
                    uint64_t output_pool_size,
                    bool enable_csv_logging,
                    const std::string& name = "ata_receiver")
        : AtaReceiverOp(ArgList{Arg("total_block", total_block),
                                Arg("partial_block", partial_block),
                                Arg("offset_block", offset_block),
                                Arg("concurrent_blocks", concurrent_blocks),
                                Arg("output_pool_size", output_pool_size),
                                Arg("enable_csv_logging", enable_csv_logging)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyDummyReceiverOp : public DummyReceiverOp {
public:
    using DummyReceiverOp::DummyReceiverOp;

    PyDummyReceiverOp(Fragment* fragment, const py::args& args,
                      const stelline::BlockShape& total_block,
                      const std::string& name = "dummy_receiver")
        : DummyReceiverOp(ArgList{Arg("total_block", total_block)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PySorterOp : public SorterOp {
public:
    using SorterOp::SorterOp;

    PySorterOp(Fragment* fragment, const py::args& args,
               uint64_t depth,
               const std::string& name = "sorter")
        : SorterOp(ArgList{Arg("depth", depth)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_transport_ops, m) {
    m.doc() = "Stelline transport operators module";

    py::class_<AtaReceiverOp, PyAtaReceiverOp, Operator, std::shared_ptr<AtaReceiverOp>>(m, "AtaReceiverOp")
        .def(py::init<Fragment*, const py::args&, const stelline::BlockShape&,
                      const stelline::BlockShape&, const stelline::BlockShape&,
                      uint64_t, uint64_t, bool, const std::string&>(),
             py::arg("fragment"),
             py::arg("total_block"),
             py::arg("partial_block"),
             py::arg("offset_block"),
             py::arg("concurrent_blocks"),
             py::arg("output_pool_size"),
             py::arg("enable_csv_logging"),
             py::arg("name") = "ata_receiver");

    py::class_<DummyReceiverOp, PyDummyReceiverOp, Operator, std::shared_ptr<DummyReceiverOp>>(m, "DummyReceiverOp")
        .def(py::init<Fragment*, const py::args&, const stelline::BlockShape&, const std::string&>(),
             py::arg("fragment"),
             py::arg("total_block"),
             py::arg("name") = "dummy_receiver");

    py::class_<SorterOp, PySorterOp, Operator, std::shared_ptr<SorterOp>>(m, "SorterOp")
        .def(py::init<Fragment*, const py::args&, uint64_t, const std::string&>(),
             py::arg("fragment"),
             py::arg("depth"),
             py::arg("name") = "sorter");

    // Expose transport constants
    m.attr("TRANSPORT_HEADER_SIZE") = AtaReceiverOp::TransportHeaderSize;
    m.attr("VOLTAGE_HEADER_SIZE") = AtaReceiverOp::VoltageHeaderSize;
    m.attr("VOLTAGE_DATA_SIZE") = AtaReceiverOp::VoltageDataSize;
}
