#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/filesystem/base.hh>

namespace py = pybind11;
using namespace stelline::operators::filesystem;
using namespace holoscan;

class PyDummyWriterOp : public DummyWriterOp {
public:
    using DummyWriterOp::DummyWriterOp;

    PyDummyWriterOp(Fragment* fragment,
                    const py::args& args,
                    const std::string& name = "dummy_writer")
        : DummyWriterOp() {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PySimpleWriterOp : public SimpleWriterOp {
public:
    using SimpleWriterOp::SimpleWriterOp;

    PySimpleWriterOp(Fragment* fragment,
                     const py::args& args,
                     const std::string& file_path,
                     const std::string& name = "simple_writer")
        : SimpleWriterOp(ArgList{Arg("file_path", file_path)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PySimpleWriterRdmaOp : public SimpleWriterRdmaOp {
public:
    using SimpleWriterRdmaOp::SimpleWriterRdmaOp;

    PySimpleWriterRdmaOp(Fragment* fragment,
                         const py::args& args,
                         const std::string& file_path,
                         const std::string& name = "simple_writer_rdma")
        : SimpleWriterRdmaOp(ArgList{Arg("file_path", file_path)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

#ifdef STELLINE_LOADER_FBH5
class PyFbh5WriterRdmaOp : public Fbh5WriterRdmaOp {
public:
    using Fbh5WriterRdmaOp::Fbh5WriterRdmaOp;

    PyFbh5WriterRdmaOp(Fragment* fragment,
                       const py::args& args,
                       const std::string& file_path,
                       const std::string& name = "fbh5_writer_rdma")
        : Fbh5WriterRdmaOp(ArgList{Arg("file_path", file_path)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};
#endif

#ifdef STELLINE_LOADER_UVH5
class PyUvh5WriterRdmaOp : public Uvh5WriterRdmaOp {
public:
    using Uvh5WriterRdmaOp::Uvh5WriterRdmaOp;

    PyUvh5WriterRdmaOp(Fragment* fragment,
                       const py::args& args,
                       const std::string& file_path,
                       const uint64_t dsp_channelization_rate,
                       const uint64_t dsp_integration_rate,
                       const std::string& name = "uvh5_writer_rdma")
        : Uvh5WriterRdmaOp(ArgList{
            Arg("file_path", file_path),
            Arg("dsp_channelization_rate", dsp_channelization_rate),
            Arg("dsp_integration_rate", dsp_integration_rate)
        }) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};
#endif

PYBIND11_MODULE(_filesystem_ops, m) {
    m.doc() = "Stelline filesystem operators module";

    py::class_<DummyWriterOp, PyDummyWriterOp, Operator, std::shared_ptr<DummyWriterOp>>(m, "DummyWriterOp")
        .def(py::init<Fragment*, const py::args&, const std::string&>(),
             py::arg("fragment"),
             py::arg("name") = "dummy_writer")
        .def("tick", &DummyWriterOp::tick)
        .def("format_metrics", &DummyWriterOp::formatMetrics)
        .def("set_manifest_provider", &DummyWriterOp::setManifestProvider)
        .def("set_metrics_provider", &DummyWriterOp::setMetricsProvider)
        .def_property_readonly("manifest", &DummyWriterOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &DummyWriterOp::metrics, py::return_value_policy::reference);

    py::class_<SimpleWriterOp, PySimpleWriterOp, Operator, std::shared_ptr<SimpleWriterOp>>(m, "SimpleWriterOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("name") = "simple_writer")
        .def("tick", &SimpleWriterOp::tick)
        .def("format_metrics", &SimpleWriterOp::formatMetrics)
        .def("set_manifest_provider", &SimpleWriterOp::setManifestProvider)
        .def("set_metrics_provider", &SimpleWriterOp::setMetricsProvider)
        .def_property_readonly("manifest", &SimpleWriterOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &SimpleWriterOp::metrics, py::return_value_policy::reference);

    py::class_<SimpleWriterRdmaOp, PySimpleWriterRdmaOp, Operator, std::shared_ptr<SimpleWriterRdmaOp>>(m, "SimpleWriterRdmaOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("name") = "simple_writer_rdma")
        .def("tick", &SimpleWriterRdmaOp::tick)
        .def("format_metrics", &SimpleWriterRdmaOp::formatMetrics)
        .def("set_manifest_provider", &SimpleWriterRdmaOp::setManifestProvider)
        .def("set_metrics_provider", &SimpleWriterRdmaOp::setMetricsProvider)
        .def_property_readonly("manifest", &SimpleWriterRdmaOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &SimpleWriterRdmaOp::metrics, py::return_value_policy::reference);

#ifdef STELLINE_LOADER_FBH5
    py::class_<Fbh5WriterRdmaOp, PyFbh5WriterRdmaOp, Operator, std::shared_ptr<Fbh5WriterRdmaOp>>(m, "Fbh5WriterRdmaOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("name") = "fbh5_writer_rdma")
        .def("tick", &Fbh5WriterRdmaOp::tick)
        .def("format_metrics", &Fbh5WriterRdmaOp::formatMetrics)
        .def("set_manifest_provider", &Fbh5WriterRdmaOp::setManifestProvider)
        .def("set_metrics_provider", &Fbh5WriterRdmaOp::setMetricsProvider)
        .def_property_readonly("manifest", &Fbh5WriterRdmaOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &Fbh5WriterRdmaOp::metrics, py::return_value_policy::reference);
#endif

#ifdef STELLINE_LOADER_UVH5
    py::class_<Uvh5WriterRdmaOp, PyUvh5WriterRdmaOp, Operator, std::shared_ptr<Uvh5WriterRdmaOp>>(m, "Uvh5WriterRdmaOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const uint64_t, const uint64_t, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("dsp_channelization_rate"),
             py::arg("dsp_integration_rate"),
             py::arg("name") = "uvh5_writer_rdma")
        .def("tick", &Uvh5WriterRdmaOp::tick)
        .def("format_metrics", &Uvh5WriterRdmaOp::formatMetrics)
        .def("set_manifest_provider", &Uvh5WriterRdmaOp::setManifestProvider)
        .def("set_metrics_provider", &Uvh5WriterRdmaOp::setMetricsProvider)
        .def_property_readonly("manifest", &Uvh5WriterRdmaOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &Uvh5WriterRdmaOp::metrics, py::return_value_policy::reference);
#endif
}
