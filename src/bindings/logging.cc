#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <sstream>

#include <holoscan/holoscan.hpp>

namespace py = pybind11;

namespace {

std::string format_message(const std::string& message, const py::args& args) {
    if (args.size() == 0) {
        return message;
    }

    std::string result = message;
    for (size_t i = 0; i < args.size(); ++i) {
        size_t pos = result.find("{}");
        if (pos != std::string::npos) {
            std::string arg_str = py::str(args[i]);
            result.replace(pos, 2, arg_str);
        }
    }
    return result;
}

void holoscan_log_info(const std::string& message, const py::args& args) {
    std::string formatted_msg = format_message(message, args);
    HOLOSCAN_LOG_INFO("{}", formatted_msg);
}

void holoscan_log_error(const std::string& message, const py::args& args) {
    std::string formatted_msg = format_message(message, args);
    HOLOSCAN_LOG_ERROR("{}", formatted_msg);
}

void holoscan_log_warn(const std::string& message, const py::args& args) {
    std::string formatted_msg = format_message(message, args);
    HOLOSCAN_LOG_WARN("{}", formatted_msg);
}

void holoscan_log_debug(const std::string& message, const py::args& args) {
    std::string formatted_msg = format_message(message, args);
    HOLOSCAN_LOG_DEBUG("{}", formatted_msg);
}

}  // namespace

PYBIND11_MODULE(_logging, m) {
    m.doc() = "HOLOSCAN logging functions for Python";

    m.def("log_info", &holoscan_log_info,
          "Log an info message using HOLOSCAN_LOG_INFO",
          py::arg("message"));

    m.def("log_error", &holoscan_log_error,
          "Log an error message using HOLOSCAN_LOG_ERROR",
          py::arg("message"));

    m.def("log_warn", &holoscan_log_warn,
          "Log a warning message using HOLOSCAN_LOG_WARN",
          py::arg("message"));

    m.def("log_debug", &holoscan_log_debug,
          "Log a debug message using HOLOSCAN_LOG_DEBUG",
          py::arg("message"));
}
