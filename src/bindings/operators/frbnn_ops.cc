#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/frbnn/base.hh>

namespace py = pybind11;
using namespace stelline::operators::frbnn;
using namespace holoscan;

class PyModelPreprocessorOp : public ModelPreprocessorOp {
public:
    using ModelPreprocessorOp::ModelPreprocessorOp;
    
    PyModelPreprocessorOp(Fragment* fragment, const py::args& args,
                          const std::string& name = "model_preprocessor")
        : ModelPreprocessorOp() {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyModelAdapterOp : public ModelAdapterOp {
public:
    using ModelAdapterOp::ModelAdapterOp;
    
    PyModelAdapterOp(Fragment* fragment, const py::args& args,
                     const std::string& name = "model_adapter")
        : ModelAdapterOp() {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyModelPostprocessorOp : public ModelPostprocessorOp {
public:
    using ModelPostprocessorOp::ModelPostprocessorOp;
    
    PyModelPostprocessorOp(Fragment* fragment, const py::args& args,
                           const std::string& name = "model_postprocessor")
        : ModelPostprocessorOp() {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PySimpleDetectionOp : public SimpleDetectionOp {
public:
    using SimpleDetectionOp::SimpleDetectionOp;
    
    PySimpleDetectionOp(Fragment* fragment, const py::args& args,
                        const std::string& csv_file_path,
                        const std::string& hits_directory,
                        const std::string& name = "simple_detection")
        : SimpleDetectionOp(ArgList{Arg("csv_file_path", csv_file_path),
                                     Arg("hits_directory", hits_directory)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_frbnn_ops, m) {
    m.doc() = "Stelline FRBNN operators module";

    py::class_<ModelPreprocessorOp, PyModelPreprocessorOp, Operator, std::shared_ptr<ModelPreprocessorOp>>(m, "ModelPreprocessorOp")
        .def(py::init<Fragment*, const py::args&, const std::string&>(),
             py::arg("fragment"),
             py::arg("name") = "model_preprocessor");

    py::class_<ModelAdapterOp, PyModelAdapterOp, Operator, std::shared_ptr<ModelAdapterOp>>(m, "ModelAdapterOp")
        .def(py::init<Fragment*, const py::args&, const std::string&>(),
             py::arg("fragment"),
             py::arg("name") = "model_adapter");

    py::class_<ModelPostprocessorOp, PyModelPostprocessorOp, Operator, std::shared_ptr<ModelPostprocessorOp>>(m, "ModelPostprocessorOp")
        .def(py::init<Fragment*, const py::args&, const std::string&>(),
             py::arg("fragment"),
             py::arg("name") = "model_postprocessor");

    py::class_<SimpleDetectionOp, PySimpleDetectionOp, Operator, std::shared_ptr<SimpleDetectionOp>>(m, "SimpleDetectionOp")
        .def(py::init<Fragment*, const py::args&, const std::string&, const std::string&, const std::string&>(),
             py::arg("fragment"),
             py::arg("csv_file_path"),
             py::arg("hits_directory"),
             py::arg("name") = "simple_detection");
}
