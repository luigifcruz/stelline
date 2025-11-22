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
                       const std::string& output_file_path,
                       const std::string& telinfo_file_path,
                       const std::string& obsantinfo_file_path,
                       const std::string& iers_file_path,
                       const std::string& name = "uvh5_writer_rdma")
        : Uvh5WriterRdmaOp(ArgList{Arg("output_file_path", output_file_path),
                                   Arg("telinfo_file_path", telinfo_file_path),
                                   Arg("obsantinfo_file_path", obsantinfo_file_path),
                                   Arg("iers_file_path", iers_file_path)}) {
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
        .def("collect_metrics_map", &DummyWriterOp::collectMetricsMap)
        .def("collect_metrics_string", &DummyWriterOp::collectMetricsString);

    py::class_<SimpleWriterOp, PySimpleWriterOp, Operator, std::shared_ptr<SimpleWriterOp>>(m, "SimpleWriterOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("name") = "simple_writer")
        .def("collect_metrics_map", &SimpleWriterOp::collectMetricsMap)
        .def("collect_metrics_string", &SimpleWriterOp::collectMetricsString);

    py::class_<SimpleWriterRdmaOp, PySimpleWriterRdmaOp, Operator, std::shared_ptr<SimpleWriterRdmaOp>>(m, "SimpleWriterRdmaOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("name") = "simple_writer_rdma")
        .def("collect_metrics_map", &SimpleWriterRdmaOp::collectMetricsMap)
        .def("collect_metrics_string", &SimpleWriterRdmaOp::collectMetricsString);

#ifdef STELLINE_LOADER_FBH5
    py::class_<Fbh5WriterRdmaOp, PyFbh5WriterRdmaOp, Operator, std::shared_ptr<Fbh5WriterRdmaOp>>(m, "Fbh5WriterRdmaOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("file_path"),
             py::arg("name") = "fbh5_writer_rdma")
        .def("collect_metrics_map", &Fbh5WriterRdmaOp::collectMetricsMap)
        .def("collect_metrics_string", &Fbh5WriterRdmaOp::collectMetricsString);
#endif

#ifdef STELLINE_LOADER_UVH5
    py::class_<Uvh5WriterRdmaOp, PyUvh5WriterRdmaOp, Operator, std::shared_ptr<Uvh5WriterRdmaOp>>(m, "Uvh5WriterRdmaOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&,
                      const std::string&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("output_file_path"),
             py::arg("telinfo_file_path"),
             py::arg("obsantinfo_file_path"),
             py::arg("iers_file_path"),
             py::arg("name") = "uvh5_writer_rdma")
        .def("collect_metrics_map", &Uvh5WriterRdmaOp::collectMetricsMap)
        .def("collect_metrics_string", &Uvh5WriterRdmaOp::collectMetricsString);
#endif
}
